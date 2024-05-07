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
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset
import multiprocessing
from typing import Tuple, List
import time
import json
import shutil
import sys
import logging

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
    print(f"finish Parsing arguments in {duration} seconds")
    return parsed


def setup_logging(log_file_path: str):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler(log_file_path, mode='a')  # File handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
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

    hf_dataset = hf_datasets.load_dataset('json',
                                        data_files=path,
                                        split=split)

    dataset = ConcatTokensDataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        bos_text=bos_text,
        eos_text=eos_text,
        no_wrap=no_wrap,
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
    logger.info(f'completed writing to file {file_number} in {duration} seconds')

# input_path: /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl
# split_dir: /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/split
def split_jsonl(input_path: str, split_dir: str, lines_per_file:int=4093):
    start_time = time.time()  # Start measuring time
    logger.info('begin to split source jsonl')

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
   
    file_number = 0
    buffer = []
    with open(input_path, 'r') as source_file:
        while True:
            # Read a block of lines
            buffer = [source_file.readline() for _ in range(lines_per_file)]
            # Check if the end of the file has been reached
            if not buffer[0]:
                break
            # Remove any empty strings that signify end of file in the last read block
            buffer = list(filter(None, buffer))
        
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
    logger.info(f'source jsonl split completed in {duration} seconds')

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

    mode = ConcatMode.CONCAT_TOKENS

    # logger.info(f'@{path_name}, tokenizer: {args.tokenizer}, n_cpus: {args.num_processes}')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
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
                               )

    end_time= time.time()
    duration, start_time = end_time - start_time, end_time
    logger.info(f'Loaded HF {path_name} dataset, took {duration} seconds')

    # Write samples
    with MDSWriter(columns=columns,
                   out=outpath,
                   compression=args.compression,
                   ) as out:
        # for sample in tqdm(dataset):
        for sample in dataset:
            out.write(sample)

    duration = time.time() - start_time
    logger.info(f'Converted {path_name} to mds/zstd, {duration} seconds')

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
    logger.info(f'Merged all mds/zstd in {duration} seconds')

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

    # e.g. all_path = "/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl" 
    all_path = args.path
    if os.path.isdir(all_path):
        # e.g. ["./en.jsonl", "./vi.jsonl", ...]
        data_files = glob(f'{all_path}/*')
    else:
        # e.g. ["/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl"]
        data_files = [all_path]

    logger.info(f'source data_files are:')
    for data_file in data_files:
        logger.info(data_file)

    logger.info(f'in Main, tokenizer: {args.tokenizer}')

    root = os.path.dirname(all_path)
    split_dir = os.path.join(root,'split')

    logger.info(f'root is {root}')
    logger.info(f'split_dir is {split_dir}')

    for data_file in data_files:
        split_jsonl(data_file, split_dir, lines_per_file=args.chunk_size)

    data_files_split = glob(f'{split_dir}/*')

    # ################################
    # 3. converting jonsl to hf dataset (not saved out)
    # 4. converting hf dataset to mdf dataset
    # results inside <root>/<split#>/, each containing a 'index.json' and a list of 'shard.XXXXX.mds.zstd'
    # - single processing ~ 1-2 days
    # - multi processing (test with 24 CPUs) ~ 1-2 hours, 22% utilization of 300 GB RAM
    # ################################

    # d_f = /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/split/11.jsonl
    arg_tuples = [(args, d_f) for d_f in data_files_split]

    pool = multiprocessing.Pool(processes=args.num_processes)
    pool.map(single_process, arg_tuples)
    pool.close()

    logger.info('conversion of all jsonl to mds/zstd completed')

    # ################################
    # 5. merge the metadata and dataset files from subdir to maindir
    # - single processing ~ 1.5 sec
    # ################################
    merge_shard_groups(args.out_root)


if __name__ == '__main__':
    main(parse_args())