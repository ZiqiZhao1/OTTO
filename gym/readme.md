
# OTTO
This is a PyTorch implementation for SIGKDD'25 paper: Offline Trajectory Optimization for Offline Reinforcement Learning

## Installation

1. Install the mujoco-py [mujoco-py repo](https://github.com/openai/mujoco-py).
2. install the dependencies with the following command

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
cd data/
python download_d4rl_datasets.py
```

## Example usage

1. Train World Transformers by using the following command

```
python train_next_state.py --env <env_name> --dataset <dataset_name> --normalization
python train_reward.py --env <env_name> --dataset <dataset_name> --normalization
```
2. Augment the data by using the following command
```
python aug_data_group.py --env ${env} \
                         --dataset_path data/<env_name>-<dataset_name>-v2.pkl \
                         --load_reward_group_path saved_model/<reward_transformer_path1> \
                                                  saved_model/<reward_transformer_path2> \
                                                  saved_model/<reward_transformer_path3> \
                                                  saved_model/<reward_transformer_path4> \
                         --load_state_group_path  saved_model/<state_transformer_path1> \
                                                  saved_model/<state_transformer_path2> \
                                                  saved_model/<state_transformer_path3> \
                                                  saved_model/<state_transformer_path4> \
                         --if_save \
                         --scale <scale_range> \
                         --normalization \
                         --aug_data_save_path <aug_data_save_path> \
                         --stop_iter <segments_number> \
                         --aug_length 50 \
                         --mode ${strategy_name} \
                         --if_add_noise
```
3. Merge the augmented data with original data by using the following command
```
python merge_dataset.py --ori_dataset data/${env}-${task}-v2.pkl \
                        --aug_dataset1 <aug_data_save_path> \
                        --reward_var_path1 <aug_data_save_path>-state_std.pkl \
                        --state_var_path1 <aug_data_save_path>-state_std.pkl \
                        --modify 1 \
                        --aug_dataset1_num <merge-step-number> \
                        --max_num 50 \
                        --temperature 0.7 \
                        --output_dataset <output_data_path>
```
Finally, a new dataset for offline RL is saved in `<output_data_path>`

## Reference
The code is implemented based on [DT](https://github.com/kzl/decision-transformer).
