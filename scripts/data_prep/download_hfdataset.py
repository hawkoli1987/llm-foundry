import os
import time
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import boto3
from smart_open import open
from botocore.config import Config
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse
import logging  # Add this import
import sys

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
# # ######################################## stackv2 small async I/O with auto resuming and loading from local parquet

# Function to download and decode content from S3
def download_content(file):
    s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
    with open(s3_url, "rb", compression=".gz", transport_params={"client": s3_client}) as fin:
        file["content"] = fin.read().decode(file["src_encoding"])
    return file

# Function to download contents of all files in the same row
def download_contents(files):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_content, file) for file in files]
        for future in as_completed(futures):
            future.result()
    return {"files": files}

# Load existing index from the index.json file
def load_existing_index(index_path):
    existing_index = set()
    if not os.path.exists(index_path):
        return existing_index
    with open(index_path, 'r') as index_file:
        for line in index_file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                existing_index.add(json.loads(line))
    return existing_index

# Append a new index entry to the index.json file
def append_to_index(index_key, index_path):
    with open(index_path, 'a') as index_file:
        json.dump(index_key, index_file)
        index_file.write('\n')

# Monitor the size of JSONL and index files and calculate download speed
def monitor_file_sizes(jsonl_path, index_path, row_count, index_set, start_time, jsonl_size_):
    end_time = time.time()
    jsonl_size = os.path.getsize(jsonl_path) / (1024 * 1024)  # Size in MB
    index_size = os.path.getsize(index_path) / (1024 * 1024)  # Size in MB
    download_speed = (jsonl_size - jsonl_size_)/ (end_time - start_time)  # Speed in MB/s
    num_entries = len(index_set)
    logger.info(f"Processed {row_count} rows")
    logger.info(f"Size of {os.path.basename(jsonl_path)}: {jsonl_size:.2f} MB")
    logger.info(f"Size of {os.path.basename(index_path)}: {index_size:.2f} MB")
    logger.info(f"Number of entries in the index set: {num_entries}")
    logger.info(f"Download speed of {os.path.basename(jsonl_path)}: {download_speed:.2f} MB/s")
    return end_time, jsonl_size

# Function to process each Parquet file
def process_parquet_file(parquet_path):
    jsonl_path = parquet_path.replace(".parquet", ".jsonl")
    index_path = parquet_path.replace(".parquet", "_index.json")
    base_name = os.path.basename(jsonl_path)

    ds = pd.read_parquet(parquet_path)
    existing_index = load_existing_index(index_path)

    # Save the dataset to a JSONL file and update the index in real-time
    with open(jsonl_path, 'a') as jsonl_file:
        row_count = 0
        start_time = time.time()
        jsonl_size = os.path.getsize(jsonl_path) / (1024 * 1024)  # Size in MB
        for row in process_rows_lazily(ds):
            row_count += 1
            # logger.info(f"{base_name} row_count: {row_count}")
            for file in row["files"]:
                index_key = f"{row['repo_url']}_{file['blob_id']}"
                if index_key in existing_index:
                    # logger.info(f"Skipping {index_key} as it already exists in the index")
                    continue
                # Save content to JSONL file
                jsonl_obj = {"text": file["content"]}
                json.dump(jsonl_obj, jsonl_file)
                jsonl_file.write('\n')
                # Append new index key to the index set and index file
                existing_index.add(index_key)
                append_to_index(index_key, index_path)
                # logger.info(f"Downloaded {index_key}")

            if row_count % 1000 == 0:
                # Monitor the file sizes and number of index entries every 200 rows
                start_time, jsonl_size = monitor_file_sizes(jsonl_path, index_path, row_count, existing_index, start_time, jsonl_size)

# Generator function to process each row lazily
def process_rows_lazily(ds):
    for idx, row in ds.iterrows():
        row['processed_files'] = download_contents(row["files"])
        yield row

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a Parquet dir.')
    parser.add_argument('parquet_dir', type=str, help='The file path of the Parquet file to process')
    parser.add_argument('n_cpu', type=int, help='Number of available CPUs')
    parser.add_argument('log_file_path', type=str, help='log path')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('aws_access_key_id:', os.environ["AWS_ACCESS_KEY_ID"])
    
    setup_logging(args.log_file_path)

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    s3_client = session.client("s3")
    # Get the list of Parquet files
    parquet_files = sorted([os.path.join(args.parquet_dir, f) for f in os.listdir(args.parquet_dir) if f.endswith(".parquet")])
    parquet_files_subset0 = parquet_files[0:100]
    parquet_files_subset6 = parquet_files[100:200]
    parquet_files_subset1 = parquet_files[200:300]
    parquet_files_subset2 = parquet_files[300:400]
    parquet_files_subset3 = parquet_files[400:500]
    parquet_files_subset4 = parquet_files[500:600]
    parquet_files_subset5 = parquet_files[600:640]


    # Number of worker processes
    # num_workers = args.n_cpu # min(cpu_count(), len(parquet_files))

    # # Create a pool of worker processes
    # with Pool(processes=num_workers) as pool:
    #     # Map the worker function to the Parquet files
    #     pool.map(process_parquet_file, parquet_files)

    num_workers = 50

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_parquet_file, f) for f in parquet_files_subset6]
        for future in as_completed(futures):
            future.result()

# python \
#     /home/project/11003280/yuli/llm-foundry/scripts/data_prep/download_hfdataset.py \
#     /home/project/11003280/data/stackv2/the-stack-v2-train-smol-ids/split \
#     8 \
#     /home/project/11003280/log/data_download/stackv2.out

    # find . -maxdepth 1 -name "*.jsonl" -exec du -ch {} + | grep total$


# # ######################################## stackv2 small async I/O with auto resuming and loading from local parquet

# # Function to download and decode content from S3
# def download_content(file):
#     s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
#     with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
#         file["content"] = fin.read().decode(file["src_encoding"])
#     return file

# # Function to download contents of all files in the same row
# def download_contents(files):
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(download_content, file) for file in files]
#         for future in as_completed(futures):
#             future.result()
#     return {"files": files}

# def process_rows_lazily(ds):
#     for idx, row in ds.iterrows():
#         row['processed_files'] = download_contents(row["files"])
#         yield row

# # Load existing index from the index.json file
# def load_existing_index(index_path):
#     existing_index = set()
#     if not os.path.exists(index_path):
#         logger.info('loaded from an empty index')
#         return existing_index
#     with open(index_path, 'r') as index_file:
#         for line in index_file:
#             line = line.strip()
#             if line:  # Ensure the line is not empty
#                 existing_index.add(json.loads(line))
#         logger.info(f'existing_index has size {len(existing_index)}')
#     return existing_index

# # Append a new index entry to the index.json file
# def append_to_index(index_key, index_path):
#     with open(index_path, 'a') as index_file:
#         json.dump(index_key, index_file)
#         index_file.write('\n')

# if __name__ == "__main__":
#     session = boto3.Session(
#         aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#         aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
#     )
#     logger.info('aws_access_key_id:', os.environ["AWS_ACCESS_KEY_ID"])
#     s3 = session.client("s3")

#     ds = pd.read_parquet("/home/project/11003280/data/stackv2/the-stack-v2-train-smol-ids/data/train-00000-of-00064.parquet")

#     # Specify the paths to save the JSONL file and index file, create the parent directory if it doesn't exist
#     jsonl_path = "/home/project/11003280/data/stackv2/train-00000-of-00064.jsonl"
#     index_path = "/home/project/11003280/data/stackv2/index_train-00000-of-00064.json"
#     os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

#     # Load existing index to avoid duplication
#     existing_index = load_existing_index(index_path)

#     # Save the dataset to a JSONL file and update the index in real-time
#     with open(jsonl_path, 'a') as jsonl_file:
#         row_count = 0
#         for row in process_rows_lazily(ds):
#             row_count += 1
#             # logger.info(f"now downloading {row['repo_name']} with {row['num_files']}")
#             logger.info(f"row_count: {row_count}")
#             for file in row["files"]:                
#                 index_key = f"{row['repo_url']}_{file['blob_id']}"
#                 if index_key in existing_index:
#                     logger.info(f"Skipping {index_key} as it already exists in the index")
#                     continue
#                 # Save content to JSONL file
#                 jsonl_obj = {"text": file["content"]}
#                 json.dump(jsonl_obj, jsonl_file)
#                 jsonl_file.write('\n')
#                 # Append new index key to the index set and index file
#                 existing_index.add(index_key)
#                 append_to_index(index_key, index_path)
#                 logger.info(f"Downloaded {index_key}")
                
#             # monitoring
#             if row_count % 200 == 0:
#                 jsonl_size = os.path.getsize(jsonl_path) / (1024**3)  # Size in GB
#                 index_size = os.path.getsize(index_path) / (1024**3)  # Size in GB
#                 logger.info(f"Processed {row_count} rows")
#                 logger.info(f"Size of {jsonl_path}: {jsonl_size:.3f} GB")
#                 logger.info(f"Size of {index_path}: {index_size:.3f} GB")

# ######################################## stackv2 small async I/O with auto resuming

# # Function to download and decode content from S3
# def download_content(file):
#     s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
#     with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
#         file["content"] = fin.read().decode(file["src_encoding"])
#     return file

# # Function to download contents of all files in the same row
# def download_contents(files):
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(download_content, file) for file in files]
#         for future in as_completed(futures):
#             future.result()
#     return {"files": files}

# # Load existing index from the index.json file
# def load_existing_index(index_path):
#     existing_index = set()
#     if not os.path.exists(index_path):
#         logger.info('loaded from an empty index')
#         return existing_index
#     with open(index_path, 'r') as index_file:
#         for line in index_file:
#             line = line.strip()
#             if line:  # Ensure the line is not empty
#                 existing_index.add(json.loads(line))
#         logger.info(f'existing_index has size {len(existing_index)}')
#     return existing_index

# # Append a new index entry to the index.json file
# def append_to_index(index_key, index_path):
#     with open(index_path, 'a') as index_file:
#         json.dump(index_key, index_file)
#         index_file.write('\n')

# if __name__ == "__main__":
#     session = boto3.Session(
#         aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#         aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
#     )
#     logger.info('aws_access_key_id:', os.environ["AWS_ACCESS_KEY_ID"])
#     s3 = session.client("s3")

#     ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True)
#     ds = ds.map(lambda row: download_contents(row["files"]))

#     # Specify the paths to save the JSONL file and index file, create the parent directory if it doesn't exist
#     jsonl_path = "/home/project/11003280/data/stackv2/dataset2.jsonl"
#     index_path = "/home/project/11003280/data/stackv2/index.json"
#     os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

#     # Load existing index to avoid duplication
#     existing_index = load_existing_index(index_path)

#     # Save the dataset to a JSONL file and update the index in real-time
#     with open(jsonl_path, 'a') as jsonl_file:
#         row_count = 0
#         for row in ds:
#             row_count += 1
#             # logger.info(f"now downloading {row['repo_name']} with {row['num_files']}")
#             logger.info(f"row_count: {row_count}")
#             for file in row["files"]:                
#                 index_key = f"{row['repo_url']}_{file['blob_id']}"
#                 if index_key in existing_index:
#                     logger.info(f"Skipping {index_key} as it already exists in the index")
#                     continue
#                 # Save content to JSONL file
#                 jsonl_obj = {"text": file["content"]}
#                 json.dump(jsonl_obj, jsonl_file)
#                 jsonl_file.write('\n')
#                 # Append new index key to the index set and index file
#                 existing_index.add(index_key)
#                 append_to_index(index_key, index_path)
#                 logger.info(f"Downloaded {index_key}")
                
#             # monitoring
#             if row_count % 200 == 0:
#                 jsonl_size = os.path.getsize(jsonl_path) / (1024**3)  # Size in GB
#                 index_size = os.path.getsize(index_path) / (1024**3)  # Size in GB
#                 logger.info(f"Processed {row_count} rows")
#                 logger.info(f"Size of {jsonl_path}: {jsonl_size:.2f} GB")
#                 logger.info(f"Size of {index_path}: {index_size:.2f} GB")

# # ######################################## stackv2 small async I/O
# import boto3
# from smart_open import open
# from botocore.config import Config
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # example of a 'file'. The column 'files' or each row in source jsonl contains multiple 'file'
#     # {
#     #     "blob_id": "b2847f05ecdb7d52b2b55b80a522b3f128c13910",
#     #     "path": "/PaginaConsultaMySQL (1).php",
#     #     "content_id": "0158c67f95db401e39e462f3a335c1b7098c6d37",
#     #     "language": "PHP",
#     #     "length_bytes": 2337,
#     #     "detected_licenses": [],
#     #     "license_type": "no_license",
#     #     "src_encoding": "UTF-8",
#     #     "is_vendor": false,
#     #     "is_generated": false,
#     #     "alphanum_fraction": 0.6399828791618347,
#     #     "alpha_fraction": 0.630993127822876,
#     #     "num_lines": 60,
#     #     "avg_line_length": 37.95000076293945,
#     #     "max_line_length": 151
#     # }
# # Function to download and decode content from S3
# def download_content(file):
#     # example of s3_url:
#     # s3://softwareheritage/content/b2847f05ecdb7d52b2b55b80a522b3f128c13910
#     s3_url = f"s3://softwareheritage/content/{file['blob_id']}"
#     with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
#         # after download and decode completes, a new 'content' field will be added to the 'file' dict
#         file["content"] = fin.read().decode(file["src_encoding"])
#     return file

# # Function to download contents of all files in the same row
# def download_contents(files):
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = [executor.submit(download_content, file) for file in files]
#         for future in as_completed(futures):
#             future.result()
#     # returns 'files' which is a dict of all 'file' in the same row
#     return {"files": files}

# if __name__ == "__main__":
#     session = boto3.Session(
#         aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#         aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
#     logger.info('aws_access_key_id:', os.environ["AWS_ACCESS_KEY_ID"])
#     s3 = session.client("s3")

#     ds = load_dataset("bigcode/the-stack-v2-train-smol-ids", split="train", streaming=True)
#     ds = ds.map(lambda row: download_contents(row["files"]))

#     # Specify the path to save the JSONL file, create the parent directory if it doesn't exist
#     jsonl_path = "/home/project/11003280/data/stackv2/dataset2.jsonl"
#     os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

#     # Save the dataset to a JSONL file
#     with open(jsonl_path, 'w') as jsonl_file:
#         for row in ds:
#         # row has dict_keys(['repo_name', 'repo_url', 'snapshot_id', 'revision_id', 'directory_id', 'branch_name', 'visit_date', 'revision_date', 'committer_date', 'github_id', 'star_events_count', 'fork_events_count', 'gha_license_id', 'gha_created_at', 'gha_updated_at', 'gha_pushed_at', 'gha_language', 'files', 'num_files']
#             logger.info(f"now downloading {row['repo_name']} with {row['num_files']}")
#             for file in row["files"]:
#                 jsonl_obj = {"text": file["content"]}
#                 json.dump(jsonl_obj, jsonl_file)
#                 jsonl_file.write('\n')



# ######################################## fineweb-edu (parquet finish)
# # use name="sample-10BT" to use the 10BT sample
# ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)

# # # peek into the content
# # for row in ds:
# #     logger.info(row["text"])
# #     break

# # Specify the path to save the JSONL file
# jsonl_path = "/home/project/11003280/data/FWedu/combined.jsonl"
# os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# # Convert and save the dataset as JSONL
# with open(jsonl_path, 'w') as jsonl_file:
#     for row in ds:
#         logger.info(f"writing id: {row['id']}")
#         json.dump(row, jsonl_file)
#         jsonl_file.write('\n')

# ######################################## open-web-math (Complete)
# from datasets import load_dataset
# ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
# # peek into the content
# for row in ds:
#     logger.info(row["text"])
#     break

# # Specify the path to save the JSONL file
# jsonl_path = "/home/project/11003280/data/openwebmath/combined.jsonl"
# os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# # Convert and save the dataset as JSONL
# with open(jsonl_path, 'w') as jsonl_file:
#     for row in ds:
#         logger.info(f"writing id: {row['url']}")
#         json.dump(row, jsonl_file)
#         jsonl_file.write('\n')

# ######################################## LoC-PD-Books-preprocessed (Complete)
# from datasets import load_dataset
# ds = load_dataset("pszemraj/LoC-PD-Books-preprocessed", split="train", streaming=True)
# # # peek into the content
# # for row in ds:
# #     logger.info(row["text"])
# #     break

# # Specify the path to save the JSONL file
# jsonl_path = "/home/project/11003280/data/LoCPD/combined.jsonl"
# os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# # Convert and save the dataset as JSONL
# with open(jsonl_path, 'w') as jsonl_file:
#     for row in ds:
#         logger.info(f"writing id: {row['title']}")
#         json.dump(row, jsonl_file)
#         jsonl_file.write('\n')

# # ######################################## StackExchange
# from datasets import load_dataset


# ds = load_dataset("togethercomputer/RedPajama-Data-1T", split="train", streaming=True)
# # # peek into the content
# # for row in ds:
# #     logger.info(row["text"])
# #     break

# # Specify the path to save the JSONL file
# jsonl_path = "/home/project/11003280/data/LoCPD/combined.jsonl"
# os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# # Convert and save the dataset as JSONL
# with open(jsonl_path, 'w') as jsonl_file:
#     for row in ds:
#         logger.info(f"writing id: {row['title']}")
#         json.dump(row, jsonl_file)
#         jsonl_file.write('\n')

# ######################################## 
# from datasets import load_dataset
# ds = load_dataset("pszemraj/LoC-PD-Books-preprocessed", split="train", streaming=True)
# # # peek into the content
# # for row in ds:
# #     logger.info(row["text"])
# #     break

# # Specify the path to save the JSONL file
# jsonl_path = "/home/project/11003280/data/LoCPD/combined.jsonl"
# os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# # Convert and save the dataset as JSONL
# with open(jsonl_path, 'w') as jsonl_file:
#     for row in ds:
#         logger.info(f"writing id: {row['title']}")
#         json.dump(row, jsonl_file)
#         jsonl_file.write('\n')


# ########################################

# # logger.info(f"Dataset saved as JSONL to {jsonl_path}")

# /home/project/11003280/vault/ngan/SEA_fm_data/arf/data/final_3B_data/mC4_dedup/zh

# /home/project/11003280/data/sealionpilev1

# cp -r 

# output_dir="/home/project/11003280/data/sealionpilev1/zh"
# mkdir -p "$output_dir"
# for i in {0..14}
# do
#     file_name=$(printf "train_%02d.jsonl" "$i")
#     if [ -f "$file_name" ]; then
#         cp "$file_name" "$output_dir"
#         echo "Copied $file_name to $output_dir"
#     else
#         echo "$file_name does not exist in the current directory"
#     fi
# done
# echo "Done."
