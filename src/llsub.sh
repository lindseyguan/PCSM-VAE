#!/bin/bash

source /etc/profile

# Do the extra step to make 'conda activate' possible inside a script
__conda_setup="$('/state/partition1/llgrid/pkg/anaconda/anaconda3-2021a/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

module load anaconda

conda deactivate
conda activate pcsm

python trainer.py --train_data train_clean.csv  \
				  --val_data val_clean.csv  \
				  --input_dim 239  \
				  --z_dim 16  \
				  --hidden_dim 64  \
				  --use_cuda  \
				  --num_epochs 50  \
				  --batch_size 32  

python evaluate.py z16_hidden64_epochs50_bs32.pt val_clean.csv
