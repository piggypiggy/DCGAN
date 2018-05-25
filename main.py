import argparse
from train import train
from test import infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        help='could be either infer or train')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='directory to save models')
    parser.add_argument('--batch_size', type=int, default='20',
                        help='train batch size')
    parser.add_argument('--epoch', type=int, default='10',
                        help='train epoch num')
    parser.add_argument('--nd', type=int, default='100',
                        help='noise dimension')
    parser.add_argument('--num', type=int, default='1',
                        help='which number to infer')
    args = parser.parse_args()

    # if not os.path.exists(args.model_dir):
    #     os.mkdir(args.model_dir)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
    else:
        print('unknown mode')