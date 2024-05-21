export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/project/11003280/yuli/nscc_working/arf/mosaicml_workspace/credential/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/py310

# Change directory to the directory containing the data preparation script
cd /home/users/nus/huangyl/scratch/code/llm-foundry/scripts/data_prep

export log_file="/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion.out"
sleep 30

echo "bash commands sent"
pwd

# # start a new log_file from afresh
# if [ -f "$log_file" ]; then
#     rm "$log_file"
# fi

python convert_dataset_json_mp.py \
    --path "/home/users/nus/huangyl/scratch/data/50B_for_Yuli/yuli_data/combined.jsonl" \
    --out_root "/home/users/nus/huangyl/scratch/data/50B_for_Yuli/out4" \
    --log_file_path "/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion.out" \
    --split "train" \
    --num_processes 64 \
    --concat_tokens 2048 \
    --tokenizer "aisingapore/sea-lion-7b" \
    --eos_text '<|endoftext|>' \
    --compression "zstd" \
    --chunk_size 300000 \
    |& tee -a "$log_file"
    # >> "$log_file" 2>&1
