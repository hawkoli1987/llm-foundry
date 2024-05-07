export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/project/11003280/yuli/nscc_working/engr/mosaicml_workspace/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/mosaic_yuli

# Change directory to the directory containing the data preparation script
cd /home/project/11003280/yuli/llm-foundry/scripts/data_prep
export log_file="/home/project/11003280/yuli/llm-foundry/log/mdsconversion.out"