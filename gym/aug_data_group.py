import copy
import json

from numpy import array

import gym
import d4rl
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from mujoco_py import MjSimState
from decision_transformer.models.my_next_state_transformer import StateTransformer
from decision_transformer.models.my_reward_transformer import RewardTransformer
import numpy as np
import torch

import argparse
import pickle
import random
import sys


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def get_sim_state(state, env_name):
    qpos = array([0.])
    # print(state[0:5])
    if env_name == 'hopper':
        gpos = np.append(qpos, state[0:5])
        qvel = state[5:]
    elif env_name == 'halfcheetah':
        gpos = np.append(qpos, state[0:8])
        qvel = state[8:]
    elif env_name == 'walker2d':
        gpos = np.append(qpos, state[0:8])
        qvel = state[8:]
    else:
        gpos = state[0:15]
        qvel = state[15:]
    return MjSimState(time=0.0, qpos=gpos, qvel=qvel, act=None, udd_state={})


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    print('\n' * 10)
    print('=' * 50)
    print('start augment')
    print('env', variant['env'])
    print('dataset_path', variant['dataset_path'])
    print('load_reward_group_path', variant['load_reward_group_path'])
    print('load_state_group_path', variant['load_state_group_path'])
    print('aug_data_save_path', variant['aug_data_save_path'])
    print('if_add_noise', variant['if_add_noise'])
    print('stop_iter', variant['stop_iter'])
    sys.stdout.flush()

    if env_name == 'hopper':
        env = gym.make('Hopper-v3', terminate_when_unhealthy=False)
        max_ep_len = 1000
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3', terminate_when_unhealthy=False)
        max_ep_len = 1000
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
    else:
        if_ant = True
        env = gym.make(env_name)
        max_ep_len = 1000

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    # print(action_low)
    # print(action_low.shape)
    action_high = env.action_space.high

    with open(variant['dataset_path'], 'rb') as f:
        trajectories = pickle.load(f)

    def get_state_mean_std(trajectories):
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        return torch.from_numpy(state_mean).to(device=device), torch.from_numpy(state_std).to(device=device)

    state_mean, state_std = get_state_mean_std(trajectories)
    if not variant['normalization']:
        state_mean, state_std = 0., 1.

    reward_model_group = []
    state_model_group = []
    reward_model_group_path = variant['load_reward_group_path']
    state_model_group_path = variant['load_state_group_path']

    print('loading model...')
    for reward_path, state_path in zip(reward_model_group_path, state_model_group_path):
        # load model
        reward_model_state = torch.load(reward_path)
        state_model_state = torch.load(state_path)
        reward_model = RewardTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant['reward_K'],
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
        state_model = StateTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=variant['state_K'],
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
        reward_model.load_state_dict(reward_model_state)
        reward_model.to(device=device)
        state_model.load_state_dict(state_model_state)
        state_model.to(device=device)
        reward_model_group.append(reward_model)
        state_model_group.append(state_model)
        print('Reward model K:', reward_model.max_length)
        print('Next State model K:', state_model.max_length)

    # reward_model.eval()
    # state_model.eval()
    print('state_mean', state_mean)
    print('state_std', state_std)

    max_k = max(reward_model_group[0].max_length, state_model_group[0].max_length)
    max_length = max_k
    if max_length < variant['start_time']:
        max_length = variant['start_time']
    # scale of disturbance
    scale = variant['scale']
    print('scale', scale)

    average_reward_loss = []
    average_next_state_loss = []
    aug_data = []
    reward_info = []
    aug_length = variant['aug_length']

    reward_loss = []
    state_loss = []
    pre_reward_std = []
    pre_state_std = []

    # if the mode is high_reward, split the original trajectories
    # sort the continuous aug_length steps trajectories by mean reward
    # if the mode is softmax, split the original trajectories
    # choose the trajectories according to the softmax probability
    split_data = []
    if 'antmaze' in env_name:
        max_length = 5
        stop_length = max_length + aug_length
        for tra in trajectories:
            if tra['rewards'][-1] == 1 and tra['rewards'].shape[0] >= stop_length:
                append_data = {'observations': tra['observations'][-stop_length:],
                               'next_observations': tra['next_observations'][-stop_length:],
                               'actions': tra['actions'][-stop_length:],
                               'rewards': tra['rewards'][-stop_length:],
                               'terminals': tra['terminals'][-stop_length:]}
                split_data.append(append_data)
        print(
            f"env:{env_name},aug_length:{aug_length},split_data:{len(split_data)},total_timesteps:{aug_length * len(split_data)}")
        if variant['mode'] == 'softmax':
            trajectories = split_data
    elif variant['mode'] == 'high_reward' or variant['mode'] == 'softmax' or variant['mode'] == 'beam_search':
        max_length = max_k
        for tra in trajectories:
            length = tra['rewards'].shape[0]
            for i in range(1, int(length / aug_length)):
                mean_reward = np.mean(tra['rewards'][i * aug_length:(i + 1) * aug_length])
                append_data = {'observations': tra['observations'][i * aug_length - max_k:(i + 1) * aug_length],
                               'next_observations': tra['next_observations'][
                                                    i * aug_length - max_k:(i + 1) * aug_length],
                               'actions': tra['actions'][i * aug_length - max_k:(i + 1) * aug_length],
                               'rewards': tra['rewards'][i * aug_length - max_k:(i + 1) * aug_length],
                               'terminals': tra['terminals'][i * aug_length - max_k:(i + 1) * aug_length],
                               'mean_reward': mean_reward}
                split_data.append(append_data)
        if variant['mode'] == 'high_reward' or variant['mode'] == 'beam_search':
            split_data_sorted = sorted(split_data, key=lambda x: x['mean_reward'], reverse=True)
            trajectories = split_data_sorted
        elif variant['mode'] == 'softmax':
            mean_rewards = [time_stamp['mean_reward'] for time_stamp in split_data]
            weights = softmax(mean_rewards)
            trajectories = random.choices(split_data, k=5 * len(split_data), weights=weights)
    elif variant['mode'] == 'normal' and variant['stop_iter'] != 100000:
        trajectories = random.choices(trajectories, k=5 * len(trajectories))

    trajectories_length = len(trajectories)
    stop_timestep = max_length + aug_length
    print('max length:', max_length)
    print('stop_timestep:', stop_timestep)

    # bug=0

    stop_iter = variant['stop_iter']
    print_per_iter = 50
    if variant['mode'] == 'beam_search':
        print_per_iter = int(print_per_iter / variant['top_k'])

    for tra_index, tra in enumerate(trajectories):
        if tra_index == stop_iter:
            print(tra_index)
            break
        if tra_index % print_per_iter == 0:
            print(tra_index, '/', trajectories_length)
            print('mean reward loss', np.average(average_reward_loss))
            print('mean next state loss', np.average(average_next_state_loss), '\n')
            sys.stdout.flush()
        if max_length > len(tra['observations']):
            continue

        if stop_timestep > len(tra['observations']):
            continue

        # randomly select start time to augment
        if variant['mode'] == 'normal' and variant['stop_iter'] != 100000:
            if len(tra['observations']) - aug_length - 1<max_k + 1:
                continue
            max_length = random.randint(max_k + 1, len(tra['observations']) - aug_length - 1)
            stop_timestep = max_length + aug_length
            print(f"tra_index={tra_index},max_length={max_length},stop_timestep={stop_timestep}")
            sys.stdout.flush()
        # set real env
        env.reset()

        states = torch.zeros((0, state_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = np.ndarray(shape=[0], dtype=np.float32)
        next_states = []
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        # init state
        cur_state = tra['observations'][0]
        states = torch.cat([states, torch.from_numpy(cur_state).to(device=device).reshape(1, state_dim)], dim=0)

        cur_reward_loss = []
        cur_state_loss = []
        cur_state_std = []
        cur_reward_std = []

        for index in range(0, max_length):
            next_states.append(tra['next_observations'][index])
            action = tra['actions'][index]
            actions = torch.cat([actions, torch.from_numpy(action).to(device=device).reshape(1, act_dim)], dim=0)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (index + 1)], dim=1)
            rewards = np.append(rewards, np.array(tra['rewards'][index]))
            states = torch.cat(
                [states, torch.from_numpy(tra['next_observations'][index]).to(device=device).reshape(1, state_dim)],
                dim=0)

        # make disturbance of action and input it to RewardTransformer and StateTransformer to augment data
        if variant['mode'] != 'beam_search':
            for index in range(max_length, stop_timestep):
                action = tra['actions'][index]
                # print('action1', action)

                if variant['if_add_noise']:
                    action = action + np.array([random.uniform(-scale, scale) for i_ in range(action.shape[0])])
                # action should belong to action_space
                action = np.where(action < action_low, action_low, action)
                action = np.where(action > action_high, action_high, action)
                # print('action2', action)

                actions = torch.cat([actions, torch.from_numpy(action).to(device=device).reshape(1, act_dim)], dim=0)

                # model group
                next_state_list = []
                reward_list = []
                for state_model, reward_model in zip(state_model_group, reward_model_group):
                    # print(bug)
                    # bug+=1
                    next_state = state_model.get_next_state(
                        states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                        actions=actions.to(dtype=torch.float32),
                        # we don't care about rewards & returns_to_go
                        rewards=actions.to(dtype=torch.float32),
                        returns_to_go=actions.to(dtype=torch.float32),
                        timesteps=timesteps.to(dtype=torch.long))
                    reward = reward_model.get_reward(states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                                                     actions=actions.to(dtype=torch.float32),
                                                     # we don't care about rewards & returns_to_go
                                                     rewards=actions.to(dtype=torch.float32),
                                                     returns_to_go=actions.to(dtype=torch.float32),
                                                     timesteps=timesteps.to(dtype=torch.long))
                    next_state_list.append(next_state)
                    reward_list.append(reward)
                stacked_next_state = torch.stack(next_state_list)
                stacked_reward = torch.stack(reward_list)
                next_state = torch.mean(stacked_next_state, dim=0)
                next_state_std = torch.mean(torch.std(stacked_next_state, dim=0))
                reward = torch.mean(stacked_reward, dim=0)
                reward_std = torch.mean(torch.std(stacked_reward, dim=0))

                next_state = next_state * state_std + state_mean
                next_state = next_state.detach().cpu().numpy()
                reward = reward.detach().cpu().numpy()

                sim_state = get_sim_state(cur_state, env_name)
                # print(sim_state)
                env.sim.set_state(sim_state)
                real_next_state, real_reward, real_done, _ = env.step(action)

                # real_next_state_nor = \
                #     (torch.from_numpy(real_next_state).to(device=device) - state_std) / state_std

                cur_reward_loss.append(np.mean(np.abs(real_reward - reward)))
                cur_state_loss.append(np.mean(np.abs(real_next_state - next_state)))
                cur_state_std.append(next_state_std.detach().cpu().numpy())
                cur_reward_std.append(reward_std.detach().cpu().numpy())

                # if real_done and index != len(tra['observations']) - 1:
                #     print('\n--------------------------quit--------------------------\n')
                #     print(index)
                #     print(real_next_state)
                #     print(real_reward)
                #     break

                cur_state = next_state
                if index != stop_timestep - 1:  # last timestep not append
                    states = torch.cat([states, torch.from_numpy(cur_state).to(device=device).reshape(1, state_dim)],
                                       dim=0)
                rewards = np.append(rewards, reward)
                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (index + 1)],
                                      dim=1)
                next_states.append(next_state)

            average_reward_loss.append(np.average(cur_reward_loss))
            average_next_state_loss.append(np.average(cur_state_loss))

            reward_loss.append(cur_reward_loss)
            state_loss.append(cur_state_loss)
            pre_reward_std.append(cur_reward_std)
            pre_state_std.append(cur_state_std)

            aug_length = min(variant['aug_length'], rewards.shape[0])
            # print(states.detach().cpu().numpy().shape)
            append_data = {'observations': states.detach().cpu().numpy()[-aug_length:],
                           'next_observations': np.array(next_states)[-aug_length:],
                           'actions': actions.detach().cpu().numpy()[-aug_length:],
                           'rewards': rewards[-aug_length:],
                           'terminals': np.array([False for _ in range(states.shape[0])], dtype=np.bool_)[-aug_length:]}
            # for k, v in append_data.items():
            #     print(k, '----', 'type:', type(append_data[k]), 'size:', append_data[k].shape)
            # sys.stdout.flush()
            aug_data.append(append_data)
            reward_info.append(rewards.sum())

        # beam search
        else:
            beam = variant['beam']
            branch = variant['branch']
            top_k = variant['top_k']
            append_data = {'observations': states,
                           'next_observations': next_states,
                           'actions': actions,
                           'rewards': rewards,
                           'sum_reward': np.sum(rewards),
                           'reward_std_list': [],
                           'state_std_list': []}
            trajectory_list = [append_data for _ in range(beam)]
            # beam search for trajectory list
            for index in range(max_length, stop_timestep):
                action = tra['actions'][index]
                tem_total_trajectories = []
                for now_trajectory in trajectory_list:
                    # random generate branch times action and get its reward
                    for branch_idx in range(branch):
                        now_actions = copy.deepcopy(now_trajectory['actions'])
                        now_states = copy.deepcopy(now_trajectory['observations'])
                        now_next_sates = copy.deepcopy(now_trajectory['next_observations'])
                        now_rewards = copy.deepcopy(now_trajectory['rewards'])
                        now_reward_std_list = copy.deepcopy(now_trajectory['reward_std_list'])
                        now_state_std_list = copy.deepcopy(now_trajectory['state_std_list'])
                        if variant['if_add_noise']:
                            action = action + np.array([random.uniform(-scale, scale) for i_ in range(action.shape[0])])

                        # action should belong to action_space
                        action = np.where(action < action_low, action_low, action)
                        action = np.where(action > action_high, action_high, action)

                        now_actions = torch.cat(
                            [now_actions, torch.from_numpy(action).to(device=device).reshape(1, act_dim)], dim=0)

                        # model group
                        next_state_list = []
                        reward_list = []
                        for state_model, reward_model in zip(state_model_group, reward_model_group):
                            next_state = state_model.get_next_state(
                                states=(now_states.to(dtype=torch.float32) - state_mean) / state_std,
                                actions=now_actions.to(dtype=torch.float32),
                                # we don't care about rewards & returns_to_go
                                rewards=now_actions.to(dtype=torch.float32),
                                returns_to_go=now_actions.to(dtype=torch.float32),
                                timesteps=timesteps.to(dtype=torch.long))
                            reward = reward_model.get_reward(
                                states=(now_states.to(dtype=torch.float32) - state_mean) / state_std,
                                actions=now_actions.to(dtype=torch.float32),
                                # we don't care about rewards & returns_to_go
                                rewards=now_actions.to(dtype=torch.float32),
                                returns_to_go=now_actions.to(dtype=torch.float32),
                                timesteps=timesteps.to(dtype=torch.long))
                            next_state_list.append(next_state)
                            reward_list.append(reward)
                        stacked_next_state = torch.stack(next_state_list)
                        stacked_reward = torch.stack(reward_list)
                        next_state = torch.mean(stacked_next_state, dim=0)
                        next_state_std = torch.mean(torch.var(stacked_next_state, dim=0))
                        reward = torch.mean(stacked_reward, dim=0)
                        reward_std = torch.mean(torch.var(stacked_reward, dim=0))

                        next_state = next_state * state_std + state_mean
                        next_state = next_state.detach().cpu().numpy()
                        reward = reward.detach().cpu().numpy()

                        now_reward_std_list.append(reward_std.detach().cpu().numpy())
                        now_state_std_list.append(next_state_std.detach().cpu().numpy())

                        cur_state = next_state
                        if index != stop_timestep - 1:  # last timestep not append
                            now_states = torch.cat(
                                [now_states, torch.from_numpy(cur_state).to(device=device).reshape(1, state_dim)],
                                dim=0)
                        now_rewards = np.append(now_rewards, reward)
                        now_next_sates.append(next_state)
                        tem_total_trajectories.append({
                            'observations': now_states,
                            'next_observations': now_next_sates,
                            'actions': now_actions,
                            'rewards': now_rewards,
                            'sum_reward': np.sum(now_rewards),
                            'reward_std_list': now_reward_std_list,
                            'state_std_list': now_state_std_list
                        })
                tem_total_trajectories_sorted = sorted(tem_total_trajectories, key=lambda x: x['sum_reward'],
                                                       reverse=True)
                trajectory_list = tem_total_trajectories_sorted[:beam]
                timesteps = torch.cat(
                    [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (index + 1)],
                    dim=1)

            final_trajectory_list = trajectory_list[:top_k]
            aug_length = min(variant['aug_length'], final_trajectory_list[0]['rewards'].shape[0])
            # append the top_k reward trajectory to augmented data
            for final_trajectory in final_trajectory_list:
                append_data = {'observations': final_trajectory['observations'].detach().cpu().numpy()[-aug_length:],
                               'next_observations': np.array(final_trajectory['next_observations'])[-aug_length:],
                               'actions': final_trajectory['actions'].detach().cpu().numpy()[-aug_length:],
                               'rewards': final_trajectory['rewards'][-aug_length:],
                               'terminals': np.array([False for _ in range(aug_length)], dtype=np.bool_),
                               'state_std_list': final_trajectory['state_std_list'][-aug_length:],
                               'reward_std_list': final_trajectory['reward_std_list'][-aug_length:]}
                # print(append_data['rewards'].sum())
                # sys.stdout.flush()
                # print('*****', append_data['state_std_list'])
                # print('*****', len(append_data['state_std_list']))
                aug_data.append(append_data)
                reward_info.append(append_data['rewards'].sum())
            # print('\n')

    reward_info = np.array(reward_info)
    print('=' * 50)
    print('augment done')
    print('mean reward loss', np.average(average_reward_loss))
    print('mean next state loss', np.average(average_next_state_loss))
    print('augment tras reward mean', np.mean(reward_info))
    print('augment tras reward std', np.std(reward_info))
    sys.stdout.flush()

    saved_path = variant['aug_data_save_path']
    if variant['if_save']:
        with open(saved_path, 'wb') as file:
            pickle.dump(aug_data, file)
        reward_loss_file = saved_path[:-4] + '-reward_loss.pkl'
        with open(reward_loss_file, 'wb') as file:
            pickle.dump(reward_loss, file)
        state_loss_file = saved_path[:-4] + '-state_loss.pkl'
        with open(state_loss_file, 'wb') as file:
            pickle.dump(state_loss, file)
        reward_std_file = saved_path[:-4] + '-reward_std.pkl'
        with open(reward_std_file, 'wb') as file:
            pickle.dump(pre_reward_std, file)
        state_std_file = saved_path[:-4] + '-state_std.pkl'
        with open(state_std_file, 'wb') as file:
            pickle.dump(pre_state_std, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')  # hopper, halfcheetah, walker2d, reacher2d
    parser.add_argument('--dataset_path', type=str, default='data/halfcheetah-expert-v2.pkl')
    parser.add_argument('--if_add_noise', action='store_true')
    parser.add_argument('--normalization', action='store_false')
    parser.add_argument('--stop_iter', type=int, default=100000)
    parser.add_argument('--if_save', action='store_true')

    # parser.add_argument('--load_reward_model_path', type=str,
    #                     default='saved_model/halfcheetah-expert-reward_0.pth')
    # parser.add_argument('--load_state_model_path', type=str,
    #                     default='saved_model/halfcheetah-expert-state_0.pth')

    parser.add_argument('--load_reward_group_path', metavar='S', type=str, nargs='+')
    parser.add_argument('--load_state_group_path', metavar='S', type=str, nargs='+')

    parser.add_argument('--aug_data_save_path', type=str,
                        default='None')
    parser.add_argument('--dataset', type=str, default='None')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--reward_K', type=int, default=20)
    parser.add_argument('--state_K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str,
                        default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=10)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--aug_length', type=int, default=1000)
    parser.add_argument('--start_time', type=int, default=-1)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', type=float, default=0.05)
    parser.add_argument('--mode', type=str, default='normal')  # normal high_reward softmax beam_search
    parser.add_argument('--beam', type=int, default=10)
    parser.add_argument('--branch', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)

    args = parser.parse_args()
    args = vars(args)
    json_str = json.dumps(args, ensure_ascii=False, indent=4)
    print(json_str)
    experiment('aug-experiment', variant=args)
