#!/bin/bash
#SBATCH --job-name=2actor_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem=55G
#SBATCH --cpus-per-task=2
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --output=outputs/actor_tuning_2.out
#SBATCH --error=outputs/actor_tuning_2.err

# list of datasets to tune
dataset_array=("actor")

# fixed settings for tuning
num_epochs=2000
runs=10
normalize_m=1
model_name="HGNAN-node"
patience=50
train_size=0.5
val_size=0.25
aggregation="neighbor"
batch_size=16
mode='train'

# hyperparameter search space
wd_values=(0.0)
lr_values=(0.001)
dropout_values=(0.0 0.5)
n_layers_values=(3 5)
hidden_channels_values=(64 128 256)
s_max_values=(1 2 3) 

echo "Starting hyperparameter tuning job..."

for data_name in "${dataset_array[@]}"; do

  echo "------------------------------------------------------------"
  echo "--- STARTING TUNING FOR DATASET: $data_name ---"
  echo "------------------------------------------------------------"

  for wd in "${wd_values[@]}"; do
    for lr in "${lr_values[@]}"; do
      for dropout in "${dropout_values[@]}"; do
        for n_layers in "${n_layers_values[@]}"; do
          for hidden_channels in "${hidden_channels_values[@]}"; do
            for s_max in "${s_max_values[@]}"; do
          
              echo "Running $data_name with: s_max=$s_max, wd=$wd, lr=$lr, dropout=$dropout, n_layers=$n_layers, hidden_channels=$hidden_channels"
              
              python main_para.py \
                --runs=$runs \
                --wd=$wd \
                --model_name=$model_name \
                --data_name=$data_name \
                --dropout=$dropout \
                --n_layers=$n_layers \
                --hidden_channels=$hidden_channels  \
                --lr=$lr \
                --s_max=$s_max \
                --num_epochs=$num_epochs \
                --early_stop=1 \
                --one_m=1 \
                --normalize_m=$normalize_m \
                --bias=1  \
                --patience=$patience \
                --batch_size=$batch_size \
                --train_size=$train_size \
                --val_size=$val_size \
                --weight \
                --aggregation=$aggregation \
                --tuning \
                --mode=$mode
            
            done
          done
        done
      done
    done
  done
  
  echo "--- FINISHED TUNING FOR DATASET: $data_name ---"

done

echo "------------------------------------------------------------"
echo "All tuning jobs completed."