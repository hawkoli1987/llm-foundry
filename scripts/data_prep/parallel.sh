export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/users/nus/huangyl/scratch/code/nscc_working/arf/mosaicml_workspace/credential/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/llmfoundry_yuli
echo "env is /home/project/11003280/envs/llmfoundry_yuli"
cd /home/project/11003280/yuli/llm-foundry/scripts/data_prep
export log_file="/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion13.out"


# # start a new log_file from afresh
# if [ -f "$log_file" ]; then
#     rm "$log_file"
# fi

python convert_dataset_json_mp.py \
    --path "/home/project/11003280/vault/ngan/2024/nscc_working/arf/document_relevance_search/data/rpv2/contents_id_format_100files" \
    --out_root "/home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_pile/" \
    --log_file_path "/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion13.out" \
    --split "train" \
    --num_processes 64 \
    --concat_tokens 2048 \
    --tokenizer "meta-llama/Meta-Llama-3-8B" \
    --bos_text "<|begin_of_text|>" \
    --eos_text "<|end_of_text|>" \
    --compression "zstd" \
    --chunk_size 300000 \
    |& tee -a "$log_file"
    # >> "$log_file" 2>&1

# python convert_dataset_json_mp.py \
#     --path "/home/users/nus/huangyl/scratch/data/3lang/raw/en_hq/combined.jsonl" \
#     --out_root "/home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_hq/" \
#     --log_file_path "/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion13.out" \
#     --num_processes 4 \
#     --concat_tokens 8192 \
#     --tokenizer "meta-llama/Meta-Llama-3-8B" \
#     --chunk_size 300000 \
#     --bos_text "<|begin_of_text|>" \
#     --eos_text "<|end_of_text|>" \
#     --compression "zstd" \
#     |& tee -a "$log_file"

