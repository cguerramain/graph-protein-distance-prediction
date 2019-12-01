import pickle
from os.path import isfile
from datetime import datetime
from src.models import AntibodyResNet
from src.cli.cli_util import get_cli_input


if __name__ == '__main__':
    def main():
        desc = ''
        args = get_cli_input(desc)
        save_file = args.save_file
        if not save_file:
            save_file = '../../saved_models/{}_{}.p'.format('Antibody_ResNet', datetime.now().strftime('%d-%m-%y_%H:%M:%S'))

        resnet = AntibodyResNet(args.h5file, num_blocks=args.num_blocks, batch_size=args.batch_size)
        if isfile(args.class_weight_file):
            print('Loading class weights from {} ...'.format(args.class_weight_file))
            class_weights = pickle.load(open(args.class_weight_file, 'rb'))
        else:
            class_weights = resnet.dataset.get_balanced_class_weights(indices=resnet.train_indices)
            pickle.dump(class_weights, open(args.class_weight_file, 'wb'))
        resnet.train(save_file=save_file, epochs=args.epochs, class_weights=class_weights, lr=args.lr)

    main()

