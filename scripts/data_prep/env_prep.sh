export HF_HOME="/home/project/11003280/vault/cache_yuli"
source /home/project/11003280/yuli/nscc_working/arf/mosaicml_workspace/credential/hf_token
module load miniforge3/23.10
source activate /home/project/11003280/envs/py310

# Change directory to the directory containing the data preparation script
cd /home/users/nus/huangyl/scratch/code/llm-foundry/scripts/data_prep

export log_file="/home/users/nus/huangyl/scratch/code/llm-foundry/log/mdsconversion.out"