import argparse
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1', default='augmented_data/augmented_trajectories.pkl')
    parser.add_argument('--dataset2', default='No')
    # parser.add_argument('--dataset1',
    #                     default='aug_walker2d-expert-random-[scale]-strategy2-stop4000-v2.pkl')
    # parser.add_argument('--dataset2', default='No')
    parser.add_argument('--dataset3', default='data/hopper-medium-v2.pkl')
    parser.add_argument('--output_data',
                        default='augmented_data/D_new.pkl')
    # parser.add_argument('--output_data', default='aug_2000_walker2d-expert-random-scale-v2.pkl')
    args = vars(parser.parse_args())
    dataset1_ori = args['dataset1']
    dataset2_ori = args['dataset2']
    dataset3 = args['dataset3']
    output_data_ori = args['output_data']
    ob1 = []
    ob2 = []
    print('-' * 10)

    if dataset1_ori != 'No':
        with open(dataset1_ori, "rb") as f:
            ob1 = pkl.load(f, encoding='latin1')
        print('dataset1:', dataset1_ori)
        print('length:', len(ob1))
        print('trajectory length:', ob1[0]['rewards'].shape)
        print('-'*10)

    if dataset2_ori != 'No':
        with open(dataset2_ori, "rb") as f:
            ob2 = pkl.load(f, encoding='latin1')
        print('dataset2:', dataset2_ori)
        print('length:', len(ob2))
        print('trajectory length:', ob2[0]['rewards'].shape)
        print('-'*10)

    with open(dataset3, "rb") as f:
        ob3 = pkl.load(f, encoding='latin1')
    print('dataset3:', dataset3)
    print('length:', len(ob3))
    print('-' * 10)

    output = ob1 + ob2 + ob3
    print('output dataset:', output_data_ori)
    print('length:', len(output))
    with open(output_data_ori, 'wb') as file:
        pkl.dump(output, file)
