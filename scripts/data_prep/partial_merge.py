"""
script for 
1. manually assemble subsuts from the split folders, using merge_shard_groups_sub()
2. manually assemble set from subsets, using merge_shard_groups_sub()

This set of scripts are 'safe' and less automated, which mean to keep all the intermediate data 
during the merging. Manual removal of all intermediate level of merged data is required after 
all merging operations. It is also slower because it is based on copying instead of moving
"""

print("begin importing modules")
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional
import multiprocessing
from typing import Tuple, List, Union
import time
import json
import shutil
import sys
import logging
# import math
print("finish importing modules")

KEYS = ['text','raw_contents','contents','raw_content','content', 'paragraph', 'paragraphs']

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

def merge_shard_groups_sub(source_dir: str, out_dir: str, subset: str) -> None:
    start_time = time.time()  # Start measuring time
    # out_dir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4
    # pattern = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/arxiv*
    pattern = os.path.join(source_dir, f'{subset}*')
    # multiple from the source, sudirs = [/home/project/11003280/data_Ngan/50B_for_Yuli/out4/arix-0001, ...]
    subdirs = sorted(glob(pattern))
    # only 1 in the target, subset_dir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/arxiv/
    subset_dir = os.path.join(out_dir, subset)

    print(0, source_dir)
    print(0, out_dir)
    print(1, subset_dir)
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)
        print(1, f'subset dir {subset_dir} exists and cleared')
    os.makedirs(subset_dir)

    for subdir in subdirs:
        print(2, subdir)

    shard_id = 0
    infos = []
    for subdir in subdirs:
        # subdir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/arix-0001/index.json
        index_filename = os.path.join(subdir, 'index.json')
        print(3, f'retrieved index.json from {subdir}')
        if not os.path.exists(index_filename):
            print(f"missing {index_filename}")
            continue
        obj = json.load(open(index_filename))
        print(4, f'loaded index.json from {subdir}')

        for info in obj['shards']:
            # update the local shard id to global shard id for non-compressed data in 'mds' format
            # old_basename = shard.00095.mds
            old_basename = info['zip_data']['basename']

            raw_file_path = os.path.join(subdir, old_basename)
            if not os.path.exists(raw_file_path):
                print(f"Missing file: {raw_file_path}")
                continue

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
            new_filename = os.path.join(subset_dir, new_basename)

            # move and rename the mds/zstd files, report error if any
            try:
                shutil.copy(old_filename, new_filename)
            except OSError as e:
                print(f"Error copying file {old_filename} to {new_filename}: {e}")
            # increment global shard_id
            shard_id += 1

            # incrementally build the content of global index.json
            infos.append(info)
        
    # create new index file inside the main dir, with the global content collected from infos
    index_filename = os.path.join(subset_dir, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)

    duration = time.time() - start_time
    print(f'Merged all mds/zstd in {duration:.1f} seconds')

def merge_shard_groups_main(source_dir: str, out_dir: str, subsets: List[str]) -> None:
    start_time = time.time()  # Start measuring time

    subdirs = []
    for subset in subsets:
        subdir = os.path.join(source_dir, f'{subset}')
        subdirs.append(subdir)
        print(0, subdir)
        if not os.path.exists(subdir):
            print(0, f'Error: subset_dir doesnot exist')
            return

    print(1, out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(1, f'output dir {out_dir} exists and cleared')
    os.makedirs(out_dir)

    shard_id = 0
    infos = []
    for subdir in subdirs:
        # subdir = /home/project/11003280/data_Ngan/50B_for_Yuli/out4/arix-0001/index.json
        index_filename = os.path.join(subdir, 'index.json')
        print(3, f'retrieved index.json from {subdir}')

        if not os.path.exists(index_filename):
            print(f"4, missing index from: {subdir}")
            continue
        obj = json.load(open(index_filename))
        print(4, f'loaded index.json from {subdir}')

        for info in obj['shards']:
            # update the local shard id to global shard id for non-compressed data in 'mds' format
            # old_basename = shard.00095.mds
            old_basename = info['zip_data']['basename']

            raw_file_path = os.path.join(subdir, old_basename)
            if not os.path.exists(raw_file_path):
                print(f"5 Missing zstd file: {old_basename}")
                continue

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

            # copy the mds/zstd files, report error if any
            try:
                shutil.copy(old_filename, new_filename)
            except OSError as e:
                print(f"6. Error copying file {old_filename} to {new_filename}: {e}")
            # increment global shard_id
            shard_id += 1

            # incrementally build the content of global index.json
            infos.append(info)
        
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
    print(f'7. Merged all mds/zstd in {duration:.1f} seconds')

if __name__ == '__main__':
    # main(parse_args())

    # source_dir='/home/users/nus/huangyl/shortcuts/data/out_gojek/dolma'
    # out_dir='/home/users/nus/huangyl/shortcuts/scratch/data/out_gojek/dolma4'
    # # subset='open-web-math-train'
    # # subset='algebraic-stack-train'
    # # subset='arxiv'
    # # subset='c4'
    # # subset='cc_news'
    # # subset='megawika'
    # # subset='reddit'
    # # subset='stackexchange'
    # # subset='starcoder'
    # # subset='tulu_flan'
    # # subset='wiki'
    # # subset='cc_en_head'
    # # subset='cc_en_middle'
    # # subset='cc_en_tail'
    # # subset='falcon'
    # merge_shard_groups_sub(source_dir=source_dir, out_dir=out_dir, subset=subset)

    
    source_dir='/home/users/nus/huangyl/shortcuts/scratch/data/out_gojek/dolma4'
    out_dir='/home/users/nus/huangyl/shortcuts/scratch/data/out_gojek/dolma5'
    subsets = [
        'open-web-math-train',
        'algebraic-stack-train',
        'arxiv',
        'c4',
        'cc_news',
        'megawika',
        'reddit',
        'stackexchange',
        'starcoder',
        'tulu_flan',
        'wiki',
        'cc_en_head',
        'cc_en_middle',
        'cc_en_tail',
        'falcon'
    ]
    merge_shard_groups_main(source_dir=source_dir, out_dir=out_dir, subsets=subsets)

