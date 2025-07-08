import argparse
import pickle
import pickle as pkl

import numpy as np


def modify_dataset(aug_dataset, max_num, temperature, reward_std_data, state_std_data):
    for index in range(len(aug_dataset)):
        sorted_data = sorted(reward_std_data[index])[-max_num:]
        arr = np.array(sorted_data)
        denominator = np.sum(np.exp(arr / temperature))
        for i in range(len(aug_dataset[index]['rewards'])):
            reward_std = reward_std_data[index][i]
            state_std = state_std_data[index][i]
            alpha = 1 - np.exp(state_std / temperature) / denominator
            aug_dataset[index]['rewards'][i] = alpha * (aug_dataset[index]['rewards'][i] - reward_std)
    return aug_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dataset', type=str, required=True)
    parser.add_argument('--aug_dataset1', type=str, required=True)
    parser.add_argument('--reward_var_path1', type=str, default='None')
    parser.add_argument('--state_var_path1', type=str, default='None')
    parser.add_argument('--aug_dataset1_num', type=int, default=-1)

    parser.add_argument('--aug_dataset2', default='None')
    parser.add_argument('--reward_var_path2', type=str, default='None')
    parser.add_argument('--state_var_path2', type=str, default='None')
    parser.add_argument('--aug_dataset2_num', type=int, default=-1)


    parser.add_argument('--modify', type=int, default=0)
    parser.add_argument('--max_num', type=int, default=10)  # reward_var_factor
    parser.add_argument('--temperature', type=float, default=0.7)  # state_var_factor
    parser.add_argument('--output_dataset', type=str, required=True)

    args = vars(parser.parse_args())
    ori_dataset_path = args['ori_dataset']
    aug_dataset1_path = args['aug_dataset1']
    output_dataset = args['output_dataset']
    max_num = args['max_num']
    temperature = args['temperature']

    with open(ori_dataset_path, 'rb') as f:
        ori_dataset = pickle.load(f)
    print(ori_dataset_path, ':', len(ori_dataset))
    with open(aug_dataset1_path, 'rb') as f:
        aug_dataset1 = pickle.load(f)
    if args['aug_dataset1_num'] != -1:
        aug_dataset1 = aug_dataset1[:args['aug_dataset1_num']]
    print(aug_dataset1_path, ':', len(aug_dataset1))

    if args['modify'] != 0:
        with open(args['reward_var_path1'], 'rb') as f:
            reward_std_data = pickle.load(f)
        with open(args['state_var_path1'], 'rb') as f:
            state_std_data = pickle.load(f)
        aug_dataset1 = modify_dataset(aug_dataset1, max_num, temperature, reward_std_data, state_std_data)

    ori_dataset += aug_dataset1

    if args['aug_dataset2'] != 'None':
        with open(args['aug_dataset2'], 'rb') as f:
            aug_dataset2 = pickle.load(f)
        if args['aug_dataset2_num'] != -1:
            aug_dataset2 = aug_dataset2[:args['aug_dataset2_num']]
        if args['modify'] != 0:
            with open(args['reward_var_path2'], 'rb') as f:
                reward_std_data = pickle.load(f)
            with open(args['state_var_path2'], 'rb') as f:
                state_std_data = pickle.load(f)
            aug_dataset2 = modify_dataset(aug_dataset2, max_num, temperature, reward_std_data, state_std_data)
        ori_dataset += aug_dataset2

    with open(output_dataset, 'wb') as file:
        pkl.dump(ori_dataset, file)

    print(output_dataset, ':', len(ori_dataset))

"""
python merge_dataset.py --ori_dataset data/hopper-medium-v2.pkl \
                        --aug_dataset1 data/new_group_aug_data/hopper/medium/softmax/scale002/4000.pkl \
                        --reward_var_path1 data/new_group_aug_data/hopper/medium/softmax/scale002/4000-reward_std.pkl \
                        --state_var_path1 data/new_group_aug_data/hopper/medium/softmax/scale002/4000-state_std.pkl \
                        --modify 1 \
                        --aug_dataset1_num 2000 \
                        --max_num 10 \
                        --temperature 0.7 \
                        --output_dataset new_group_aug_dataset/hopper/medium/softmax/scale002/2000-max_10-temperature_07.pkl
                        
"""
