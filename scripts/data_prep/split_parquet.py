import os
import pandas as pd
from multiprocessing import Pool, cpu_count

split_dir = '/home/project/11003280/data/stackv2/the-stack-v2-train-smol-ids/split'
# Function to split and save a parquet file
def split_parquet_file(file_path):
    print('file_path:', file_path)

    # Get the base filename without extension
    base_filename = os.path.basename(file_path).replace('.parquet', '')
    print('base_name:', base_filename)
    
    # Check the number of existing files with the base filename prefix
    existing_files = [f for f in os.listdir(split_dir) if f.startswith(base_filename) and f.endswith('.parquet')]
    if len(existing_files) == 10:
        print(f"Skipping {file_path}, all split files already exist.")
        return

    # Load the parquet file
    df = pd.read_parquet(file_path)
    print('read parquet completed')
    
    # Calculate the number of rows per split
    num_splits = 10
    rows_per_split = len(df) // num_splits

    
    # Split the dataframe and save each part
    for i in range(num_splits):
        start_row = i * rows_per_split
        end_row = start_row + rows_per_split
        split_df = df.iloc[start_row:end_row]
        
        # Handle the last split to include any remaining rows
        if i == num_splits - 1:
            split_df = df.iloc[start_row:]
        
        # Save the split dataframe to the split directory
        split_filename = f"{base_filename}_{i}.parquet"
        split_filepath = os.path.join(split_dir, split_filename)
        if os.path.exists(split_filepath):
            print('skipping', split_filepath)
            continue
        split_df.to_parquet(split_filepath)
        print(f"Saved split file: {split_filepath}")

if __name__ == "__main__":
    # Define the directories
    data_dir = '/home/project/11003280/data/stackv2/the-stack-v2-train-smol-ids/data'

    # Create the split directory if it doesn't exist
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    # Get the list of .parquet files in the data directory
    parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]

    for file in parquet_files:
        split_parquet_file(file)

    # num_workers=8

    # # Create a pool of worker processes
    # with Pool(processes=num_workers) as pool:
    #     # Map the worker function to the Parquet files
    #     pool.map(split_parquet_file, parquet_files)

