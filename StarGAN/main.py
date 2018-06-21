import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    data_loader, data_size, c_dim = get_loader(config.data_dir, config.selected_attrs, config.split, config.batch_size,
                                               config.mode, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader, config, data_size, c_dim)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--g_model', type=int, default=1, choices=[1, 2, 3, 4], help='choose generator model')
    parser.add_argument(
        '--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')  # Was 64
    parser.add_argument(
        '--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')  # Was 64
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')  # was 16
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')  # was 0.0001
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')  # was 0.0001
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument(
        '--selected_attrs', '--list', nargs='+', help='selected attributes for the dataset', default=None)
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--split', type=float, default=0.8, help='split training and test set')
    parser.add_argument('--num_workers', type=int, default=0)  # was 1
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--force_cpu', action='store_true')

    # Directories.
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--log_dir', type=str, default='./stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='./stargan/models')
    parser.add_argument('--sample_dir', type=str, default='./stargan/samples')
    parser.add_argument('--result_dir', type=str, default='./stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
