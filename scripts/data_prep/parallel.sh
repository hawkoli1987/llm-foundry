export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/users/nus/huangyl/scratch/code/nscc_working/arf/mosaicml_workspace/credential/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/llmfoundry_yuli
echo "env is /home/project/11003280/envs/llmfoundry_yuli"
cd /home/users/nus/huangyl/scratch/code/llm-foundry/scripts/data_prep
export log_file="/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion.out"


# # start a new log_file from afresh
# if [ -f "$log_file" ]; then
#     rm "$log_file"
# fi

# python convert_dataset_json_mp.py \
#     --path "/home/project/11003280/vault/ngan/2024/nscc_working/arf/document_relevance_search/data/rpv2/contents_id_format_100files" \
#     --out_root "/home/project/11003280/vault/ngan/2024/nscc_working/arf/document_relevance_search/data/rpv2/contents_id_format_100files_out" \
#     --log_file_path "/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion.out" \
#     --split "train" \
#     --num_processes 64 \
#     --concat_tokens 2048 \
#     --tokenizer "/home/users/nus/huangyl/scratch/model/seatokenizer_128k/128000_wnorm_wodummyprefix.model" \
#     --eos_text '<|endoftext|>' \
#     --compression "zstd" \
#     --chunk_size 300000 \
#     |& tee -a "$log_file"
#     # >> "$log_file" 2>&1

python convert_dataset_json_mp.py \
    --path "/home/users/nus/huangyl/scratch/data/3lang/raw/en_hq/combined.jsonl" \
    --out_root "/home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_hq/" \
    --log_file_path "/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion13.out" \
    --num_processes 4 \
    --concat_tokens 8192 \
    --tokenizer "meta-llama/Meta-Llama-3-8B" \
    --chunk_size 300000 \
    --bos_text "<|begin_of_text|>" \
    --eos_text "<|end_of_text|>" \
    --compression "zstd" \
    |& tee -a "$log_file"


# aws s3 sync \
#     /home/users/nus/huangyl/scratch/data/3lang/out/en_hq \
#     s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/en_hq \
#     --endpoint-url https://s3-accelerate.amazonaws.com \
#     --dryrun


# cd /home/users/nus/huangyl/scratch/data/3lang/raw/en_hq/split
# find . -type f -name "*.preprocessed" -exec rm {} \;