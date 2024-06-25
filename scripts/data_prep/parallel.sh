export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/users/nus/huangyl/scratch/code/nscc_working/arf/mosaicml_workspace/credential/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/llmfoundry_yuli
echo "env is /home/project/11003280/envs/llmfoundry_yuli"
cd /home/users/nus/huangyl/scratch/code/llm-foundry/scripts/data_prep
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


aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_pile \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms_llama3/en_pile \
    --endpoint-url https://s3-accelerate.amazonaws.com \
    --dryrun


cd /home/project/11003280/vault/ngan/2024/nscc_working/arf/document_relevance_search/data/rpv2/contents_id_format_100files \
cd /home/users/nus/huangyl/scratch/data/3lang/raw/en_hq/split \
cd /home/users/nus/huangyl/scratch/data/3lang/raw/id_pile1/split
cd /home/users/nus/huangyl/scratch/data/3lang/raw/id_pile2/split
cd /home/users/nus/huangyl/scratch/data/3lang/raw/ms_hq/split
cd /home/users/nus/huangyl/scratch/data/3lang/raw/id_hq/split
cd /home/users/nus/huangyl/scratch/data/3lang/raw/ms_pile/split

ls
find . -type f -name "*.preprocessed" -exec rm {} \;
ls


# cd /home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_pile/

cp -r \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_hq/7. \
    /home/project/11003280/vault/data/sample_yuli/7.

cat *.jsonl > combined.jsonl

cp \
    /home/users/nus/huangyl/scratch/code/llm-foundry/scripts/data_prep/convert_dataset_json_mp.py \
    /home/project/11003280/vault/data/mds_conv_artifacts/llama3/

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/en_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com \
    --dryrun

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/ms_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/ms_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com


aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/ms_pile \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/ms_pile \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/id_pile2 \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/id_pile2 \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/id_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/id_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama3/en_pile \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama3/en_pile2 \
    --endpoint-url https://s3-accelerate.amazonaws.com \
    --dryrun


##################################

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/en_pile \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/en_pile \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/ms_pile \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/ms_pile \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/ms_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/ms_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/id_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/id_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/id_pile1 \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/id_pile1 \
    --endpoint-url https://s3-accelerate.amazonaws.com

##################################

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/en_hq \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/en_hq \
    --endpoint-url https://s3-accelerate.amazonaws.com

aws s3 sync \
    /home/users/nus/huangyl/scratch/data/3lang/out_llama2/id_pile2 \
    s3://seafm-cluster-2-common/CPT_experiments/50B_en_id_ms/llama2/id_pile2 \
    --endpoint-url https://s3-accelerate.amazonaws.com


cd /home/users/nus/huangyl/scratch/data/3lang/raw/id_hq/idwiki_dedup.jsonl

for file in *.jsonl; do
  # Check if there are any .jsonl files
  if [[ -f "$file" ]]; then
    # Get the number of columns (keys) in the first row of the file
    num_columns=$(head -n 1 "$file" | jq 'keys | length')
    # Print the filename and the number of columns
    echo "File: $file - Columns: $num_columns"
  else
    echo "No .jsonl files found in the current directory."
    break
  fi
done

cp -r \
    /home/users/nus/huangyl/scratch/code/llm-foundry  \
    /home/project/11003280/yuli/llm-foundry