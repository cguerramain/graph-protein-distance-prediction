import argparse
import pickle
from os.path import isfile
from datetime import datetime
from resnet_model import AntibodyGraphResNet


def get_cli_input(desc):
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--h5file', type=str, default='../data/ab_pdbs.h5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_blocks', nargs='*', type=int, default=[10])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--class_weight_file', type=str, default='../data/antibody_weights.p',
                        help='A pickle file of a torch Tensor to be used as class weights during training')

    return parser.parse_args()


if __name__ == '__main__':
    def main():
        desc = ''
        args = get_cli_input(desc)
        save_file = args.save_file
        if not save_file:
            save_file = '../saved_models/{}_{}.p'.format('Antibody_GCNN', datetime.now().strftime('%d-%m-%y_%H:%M:%S'))

        resnet = AntibodyGraphResNet(args.h5file, num_blocks=args.num_blocks, batch_size=args.batch_size)
        if isfile(args.class_weight_file):
            print('Loading class weights from {} ...'.format(args.class_weight_file))
            class_weights = pickle.load(open(args.class_weight_file, 'rb'))
        else:
            class_weights = resnet.dataset.get_balanced_class_weights(indices=resnet.train_indices)
            pickle.dump(class_weights, open(args.class_weight_file, 'wb'))
        resnet.train(save_file=save_file, epochs=args.epochs, class_weights=class_weights, lr=args.lr)

    main()

