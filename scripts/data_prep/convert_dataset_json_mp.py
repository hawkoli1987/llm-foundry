# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""

print("begin importing modules")
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, GemmaTokenizer

# when using BPE tokenizers
from llmfoundry.data import ConcatTokensDataset

# when using sentencepiece tokenizers
# from data_N import ConcatTokensDataset
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor

import multiprocessing
from typing import Tuple, List, Union
import time
import json
import shutil
import sys
import logging
# import math

KEYS = ['text','raw_contents','contents','raw_content','content']


print("finish importing modules")

class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args() -> Namespace:

    print("begin Parsing arguments")
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--log_file_path', type=str, default='./log')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')
    parser.add_argument('--split', type=str, default='train')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--chunk_size', type=int, default=int(1e6))  # Number of lines per split jsonl file
    parser.add_argument('--use_lang_id', default=False, action='store_true')
    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''

    start_time = time.time()    
    duration = time.time() - start_time
    print(f"finish parsing arguments in {duration:.1f} seconds")   
    return parsed


def setup_logging(log_file_path: str):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    # f_handler = logging.FileHandler(log_file_path, mode='a')  # File handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    # f_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    # logger.addHandler(f_handler)

    return logger


# create temp jsonl to ensure only
def preprocess_jsonl(path: str) -> str:
    temp_path = f"{path}.preprocessed"
    if not os.path.exists(temp_path):
        with open(path, 'r') as infile, open(temp_path, 'w') as outfile:
            for line in infile:
                obj = json.loads(line)
                for content_key in KEYS:
                    if content_key in obj:
                        # Ensure only 'content' field is used
                        new_obj = {content_key: obj[content_key]}
                        outfile.write(json.dumps(new_obj) + '\n')
                        break
                else:
                    logger.warning(f"Skipping object without any of the keys: {obj}")
    return temp_path

def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: Union[SentencePieceProcessor, PreTrainedTokenizerBase] = None,
    use_lang_id=False
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    preprocessed_path = preprocess_jsonl(path)

    hf_dataset = hf_datasets.load_dataset('json',
                                        data_files=preprocessed_path,
                                        split=split)

    dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
        # use_lang_id=use_lang_id # only when using SentencePiece (128k) tokenizer
    )

    return dataset


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}

# Define a worker function for multiprocessing
def process_chunk(chunk: Tuple[int, List[str], str]):
    start_time = time.time()    
    file_number, buffer, split_dir = chunk
    logger.info(f'begin writing to file {file_number}')
    with open(os.path.join(split_dir, f'{file_number}.jsonl'), 'w') as output_file:
        output_file.writelines(buffer)
    duration = time.time() - start_time
    logger.info(f'completed writing to file {file_number} in {duration:.1f} seconds')

def avg_text_length(buffer_header: list) -> float:
    if not buffer_header:
        # Return a small value if the input list is empty to prevent 0-division error
        return 1e-4

    text_lengths = []
    for line in buffer_header:       
        try:
            # Load the JSON object
            item = json.loads(line)
            for content_key in KEYS:
                if content_key in item:
                    content = item[content_key]
                    break
            else:
                raise KeyError(f"Sample does not contain any of the expected keys: {KEYS}")

            # Print the item
            text_lengths.append(len(content))
        except Exception as e:
            logger.info(f'encountered error {e}')
            logger.info(f'line is: {line}')
    if text_lengths == []:
        return 1e-4
    avg_length =sum(text_lengths)/len(text_lengths)
    return(avg_length)

# input_path: /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl
# split_dir: /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/split
def split_jsonl(input_path: str, split_dir: str, lines_per_file:int):
    start_time = time.time()  # Start measuring time
    logger.info('begin to split source jsonl')

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
   
    file_number = 0
    total_lines = 0
    buffer = []
    # default text length per sample is 2000
    std_text_length = 2e3
    sample_size = 10

    with open(input_path, 'r') as source_file:
       
        while True:
            ##################################
            # variational line number for each jsonl based on text length in each line
            # Read a block of lines            
            # buffer_header = [source_file.readline() for _ in range(sample_size)]
            # # Check if the end of the file has been reached
            # if (not buffer_header) or (not buffer_header[0]):
            #     logger.info(f'file reading completed, total {total_lines} text lines in the dataset')
            #     break

            # text_length_ratio = std_text_length/avg_text_length(buffer_header) # [2, 1, 0.1,..., 0.0001 ]
            # # low-capped the reducation rate to 0.5 == up-capped the expansion rate to 2.0
            # reduction_rate = max((1/text_length_ratio) ** 0.8, 0.5)
            # true_lines_per_file = int(lines_per_file / reduction_rate)
            # logger.info(f'split #{file_number} reduced by {reduction_rate:.1f}X >> lines/file {true_lines_per_file}')

            # buffer = buffer_header + [source_file.readline() for _ in range(true_lines_per_file-sample_size)]
           
            ##################################
            # constant line number
            # Check if the end of the file has been reached
            buffer = [source_file.readline() for _ in range(lines_per_file)]
            if (not buffer) or (not buffer[0]):
                logger.info(f'file reading completed, total {total_lines} text lines in the dataset')
                break
            
            ##################################

            # Remove any empty strings that signify end of file in the last read block
            buffer = list(filter(None, buffer))
            logger.info(f'it contains line #{total_lines} to #{total_lines+len(buffer)}')
            total_lines += len(buffer)

            # # Write to file
            # using single processing
            with open(os.path.join(split_dir, f'{file_number}.jsonl'), 'w') as output_file:
                logger.info(f'begin writing {file_number}.jsonl')
                output_file.writelines(buffer)
                logger.info(f'finish writing {file_number}.jsonl')
        
            # (didn't work) 
            # Use multiprocessing to write the buffer to file, assuming bottleneck is disk I/O
            # with multiprocessing.Pool() as pool:
            #     pool.apply_async(process_chunk, args=(file_number, buffer, split_dir))

            file_number += 1
    duration = time.time() - start_time
    logger.info(f'source jsonl split completed in {duration:.1f} seconds')

###############################
# Ngan's MP implementation
###############################
def single_process(tuple_args: Tuple[Namespace, str]) -> None:
    start_time = time.time()  # Start measuring time

    # path = /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/split/11.jsonl
    args, path = tuple_args
    # e.g. path_name = 11.
    path_name = path.split('/')[-1][:-5]
    # e.g. /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/out/11.
    outpath = os.path.join(args.out_root, path_name)

    logger.info(f"Starting processing of {path_name}")

    # Check if the previous conversion was successful, if not, reconvert
    if os.path.exists(outpath):
        if os.path.exists(os.path.join(outpath, 'index.json')):
            logger.info(f"Skipping {path_name}. Conversion already completed.")
            return
        else:
            logger.info(f"Previous conversion to {path_name} incomplete. Reconvert.")
            shutil.rmtree(outpath)

    try:
        mode = ConcatMode.CONCAT_TOKENS

        # logger.info(f'@{path_name}, tokenizer: {args.tokenizer}, n_cpus: {args.num_processes}')

        ## when using huggingface tokenizers
        # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        ## when using sentencepiece tokenizers
        # tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

        ## when using BPE dropout for Ngan
        tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                use_fast=False,
                sp_model_kwargs={'enable_sampling': True, 'nbest_size': -1, 'alpha': 0.1}
                )
        
        tokenizer.model_max_length = int(1e30)

        columns = {'tokens': 'bytes'}
    
        # Get samples
        dataset = build_hf_dataset(path=path,
                                split=args.split,
                                mode=mode,
                                max_length=args.concat_tokens,
                                bos_text=args.bos_text,
                                eos_text=args.eos_text,
                                no_wrap=args.no_wrap,
                                tokenizer=tokenizer,
                                use_lang_id=args.use_lang_id
                                )

        end_time= time.time()
        duration, start_time = end_time - start_time, end_time
        logger.info(f'Loaded HF {path_name} dataset, took {duration:.1f} seconds')

        # Write samples
        with MDSWriter(columns=columns,
                    out=outpath,
                    compression=args.compression,
                    ) as out:
            # for sample in tqdm(dataset):
            for sample in dataset:
                out.write(sample)

        duration = time.time() - start_time
        logger.info(f'Converted {path_name} to mds/zstd, {duration:.1f} seconds')
        
    except Exception as e:
        logger.error(f"Error processing {path_name}: {e}")    

# helper function to obtain global shard_id for each mds/zstd dataset
def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split('.')
    parts[1] = f'{shard_id:05}'
    return '.'.join(parts)

# move the shards (mds/zstd) from each subdir into the maindir, 
# and merge the metadata (index.json) into one global metadata
def merge_shard_groups(out_dir: str) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        out_dir (str): out_dir directory.
    """
    start_time = time.time()  # Start measuring time
    # out_dir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4
    # pattern = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/*
    pattern = os.path.join(out_dir, '*')
    # sudirs = [/home/project/11003280/data_Ngan/50B_for_Yuli/out4/0., ...]
    subdirs = sorted(glob(pattern))
    shard_id = 0
    infos = []
    for subdir in subdirs:
        # subdir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/0./index.json
        index_filename = os.path.join(subdir, 'index.json')
        obj = json.load(open(index_filename))

        # e.g. info =
        # {
        #     "column_encodings": [
        #         "bytes"
        #     ],
        #     "column_names": [
        #         "tokens"
        #     ],
        #     "column_sizes": [
        #         null
        #     ],
        #     "compression": "zstd",
        #     "format": "mds",
        #     "hashes": [],
        #     "raw_data": {
        #         "basename": "shard.00095.mds",
        #         "bytes": 57011559,
        #         "hashes": {}
        #     },
        #     "samples": 3478,
        #     "size_limit": 67108864,
        #     "version": 2,
        #     "zip_data": {
        #         "basename": "shard.00095.mds.zstd",
        #         "bytes": 12780022,
        #         "hashes": {}
        #     }
        # }


        for info in obj['shards']:
            # update the local shard id to global shard id for non-compressed data in 'mds' format
            # old_basename = shard.00095.mds
            old_basename = info['raw_data']['basename']
            # new_basename = shard.00176.mds, the shard_id is now the global shard id
            new_basename = with_id(old_basename, shard_id)
            info['raw_data']['basename'] = new_basename

            # repeat the same for the compressed data in 'zstd' format
            old_basename = info['zip_data']['basename']
            new_basename = with_id(old_basename, shard_id)
            info['zip_data']['basename'] = new_basename

            # NOTE: implicitly check if compressed file is used, if so, moves the compressed files only
            # old_filename = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/0./shard.00095.mds.zstd
            old_filename = os.path.join(subdir, old_basename)
            # new_filename = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/shard.00176.mds.zstd
            new_filename = os.path.join(out_dir, new_basename)

            # move and rename the mds/zstd files, report error if any
            assert not os.rename(old_filename, new_filename)

            # increment global shard_id
            shard_id += 1

            # incrementally build the content of global index.json
            infos.append(info)

        # remove the original index file inside each subdir, report error if any
        assert not os.remove(index_filename)
        # remove the original subdir, report error if any
        # assert not os.rmdir(subdir)
        try:
            shutil.rmtree(subdir)
        except Exception as e:
            logger.info(f"Error while removing directory {subdir}: {e}")
        
    # create new index file inside the main dir, with the global content collected from infos
    index_filename = os.path.join(out_dir, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)

    duration = time.time() - start_time
    logger.info(f'Merged all mds/zstd in {duration:.1f} seconds')

def main(args: Namespace) -> None:
    # """Main: create C4/pile streaming dataset.

    # Args:
    #     args (Namespace): Commandline arguments.
    # """
    # ################################
    # 1. combined source file is already obtained via following bash command
    # cat *.jsonl > combined.jsonl
    # ################################

    # ################################
    # 2. split the jsonl files by lines, currently used 1 million lines per split, 
    # resulted in 32 splits, inside '.<root>/split' directory, 2 options available:
    # - single processing ~ 10 mins
    # - multi processing ~ to be tested out
    # ################################

    setup_logging(args.log_file_path)

    logger.info("Input arguments:")
    for attr, value in vars(args).items():
        logger.info(f"{attr}: {value}")

    # (mode 1) directly passing in the directory containing splitted jsonl 
    # e.g args.path = "/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data" 
    if os.path.isdir(args.path):
        logger.info(f"processing from already splitted json inside {args.path}")
        split_dir = args.path

    # (mode 2) e.g. "/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl"
    else:
        logger.info(f"processing from a combined jsonl: {args.path}")
        root = os.path.dirname(args.path)
        logger.info(f'root is {root}')
        split_dir = os.path.join(root,'split')
        logger.info(f'split_dir is {split_dir}')

        # split the jsonl if not already split
        if not os.path.exists(split_dir):
            # resplit the data
            logger.info(f"splitting starts {args.path}")
            split_jsonl(args.path, split_dir, lines_per_file=args.chunk_size)
            logger.info(f"splitting completed {args.path}")
        else:
            logger.info(f'split_dir already exists, no splitting required')

    # gather all split data_files
    data_files_split = glob(os.path.join(split_dir, '*.jsonl'))
    logger.info(f'source splited jsonl files are:')
    for split_file in data_files_split:
        logger.info(split_file) 

    # ################################
    # 3. converting jonsl to hf dataset (saved to HF cache dir)
    # 4. converting hf dataset to mdf dataset
    # results inside <root>/<split#>/, each containing a 'index.json' and a list of 'shard.XXXXX.mds.zstd'
    # - single processing ~ 1-2 days
    # - multi processing (test with 24 CPUs) ~ 1-2 hours, 22% utilization of 300 GB RAM
    # ################################

    # d_f = /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/split/11.jsonl
    arg_tuples = [(args, d_f) for d_f in data_files_split]

    # clear the out_root if exists, or create one if it doesn't
    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)
        logger.info(f'output dir {args.out_root} exists and cleared')
    os.makedirs(args.out_root)

    pool = multiprocessing.Pool(processes=args.num_processes)
    pool.map(single_process, arg_tuples)
    pool.close()
    pool.join()

    logger.info('conversion of all jsonl to mds/zstd completed')

    # ################################
    # 5. merge the metadata and dataset files from subdir to maindir
    # - single processing ~ 1.5 sec
    # ################################
    merge_shard_groups(args.out_root)

    logger.info('whole conversion pipeline completed')


if __name__ == '__main__':
    main(parse_args())
    # args = parse_args()
    # logger.info(args)
    # raise ValueError("This is a custom error message")
