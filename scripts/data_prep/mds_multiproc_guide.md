# Data conservsion pipeline optimized with multi-processing

### request compute resource in NSCC
- requesting a compute node, with recommend 48 CPU and 100GB RAM (roughly need 2 hours to convert 50B token dataset)
    ```bash
    # run the job in NSCC compute node "normal queue" interactively, without specifying the script upfront
    qsub -P 11003280 -q normal -I -l select=1:ncpus=48:mem=200gb -l walltime=12:00:00
    qsub -P 11003280 -q normal -I -l select=1:ncpus=50:mem=200gb -l walltime=4:00:00
    qsub -P 11003280 -q normal -I -l select=1:ncpus=2:mem=100gb -l walltime=10:00:00

    # it will automatically directs the current terminal from the head node to the compute node
    ```
- (optional) using another terminal to ssh into the same compute node for resource monitoring, need to clone my LLM-foundry repo which contains the ssh script
    ```bash
    # change to LLM-foundry code repo
    cd /home/project/11003280/yuli/llm-foundry/scripts
    chmod +x ssh_current_job.sh
    ./ssh_current_job.sh
    ```

### inside the compute node, prepare the environment by running the script
- set the HF cache directory, HF token, log_file path as environment variables
- load the required modules
- activate the conda env (need to setup as a prerequisite)

    ```bash
    # change to LLM-foundry code repo scripts directory
    cd /home/project/11003280/yuli/llm-foundry/scripts

    chmod +x ./data_prep/env_prep.sh

    # run the environment preparation script
    ./data_prep/env_prep.shcp 
    ```

### prepare the source files in jsonl
- the source files are typically in en.jsonl, id.jsonl, etc.
- store them into one data dir <root>, e.g. '/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data'
- using the following script to combined into one large jsonl file:
    ```bash
    # change to 'root' of data directory
    cd /home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data

    # merge into one jsonl file
    cat *.jsonl > combined.jsonl
    ```

### configure the bash script
- inside '<code_repo_dir>/scripts/parallel.sh', set the following command line arguments
    ```bash
    python convert_dataset_json_mp.py \
        # path of source file
        --path "/home/project/11003280/data_Ngan/50B_for_Yuli/yuli_data/combined.jsonl" \
        # directory of output (mds/zstd) dataset file
        --out_root "/home/project/11003280/data_Ngan/50B_for_Yuli/out5" \
        # path of log
        --log_file_path "/home/project/11003280/yuli/llm-foundry/log/mdsconversion.out" \
        # a suffix labelling the 'train', 'test' and 'val' split
        --split "train" \
        # CPUs available for multi-processing
        --num_processes 48 \
        # sequence length
        --concat_tokens 2048 \
        # Huggingface tokenizer used for tokenizing the samples from text to tokens
        --tokenizer "aisingapore/sea-lion-7b" \
        # (Optional) start-of-sequence and end-of-sequence token
        --eos_text '<|endoftext|>' \
        # compression or not. If compression used, there might be more CPU load during training which can be offset with more number of dataloader workers for each GPU, assume data-parallel is used.
        --compression "zstd"
    ```

- split the combined.jsonl into smaller chunks of .jsonl files inside '<root>/split' directory, e.g. '0.jsonl', '1.jsonl', etc.
- convert each .jsonl into a HF dataset (in RAM)
- convert each HF dataset into mds/zstd file, saved in shards. 
    - each HF dataset typically is converted to one subdir (e.g. '<out_root>/0.'), 
    - each subdir will contain one metadata (i.e. 'index.json') 
    - each subdir will contain and multiple shards with local indices. (e.g. '<out_root>/11./shard.00000.mds.zstd', '<out_root>/11./shard.00130.mds.zstd', etc.)
- merge all shards from each subdir into the maindir <out_root>, 
    - reindex each shard using global shard index, and move the reindex shard to maindir,
    - merge all update metadata in each subdir into one metadata in the maindir,
    - remove all subfolders


    ```bash
    # change to LLM-foundry code repo scripts directory
    cd /home/project/11003280/yuli/llm-foundry/scripts

    # grant permission <code_repo_dir>/scripts/parallel.sh
    chmod +x ./parallel.sh

    # run the python script
    ./parallel.sh
    ```

