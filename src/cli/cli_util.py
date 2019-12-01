import argparse


def get_cli_input(desc):
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--h5file', type=str, default='../../data/ab_pdbs.h5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_blocks', nargs='*', type=int, default=[10])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--class_weight_file', type=str, default='../../data/antibody_weights.p',
                        help='A pickle file of a torch Tensor to be used as class weights during training')

    return parser.parse_args()
