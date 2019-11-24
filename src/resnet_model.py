import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from layers import ResNet2D, ResBlock2D, Edge2EdgeResNet
from data import H5AntibodyDataset
from trainer import train_and_validate


class AntibodyResNet(nn.Module):
    def __init__(self, h5file, num_blocks=10, batch_size=32, init_channels=64, num_workers=4, block=ResBlock2D):
        super(AntibodyResNet, self).__init__()
        self.dataset = H5AntibodyDataset(h5file)
        indices = list(range(len(self.dataset)))
        self.train_indices, self.test_indices = indices[len(indices) // 10:], indices[:len(indices) // 10]
        self.train_loader = data.DataLoader(self.dataset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(self.train_indices),
                                            num_workers=num_workers)
        self.test_loader = data.DataLoader(self.dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(self.test_indices),
                                           num_workers=num_workers)
        self.init_channels = init_channels
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]

        feature, _ = self.dataset[0]
        self.in_channels = feature.shape[0]
        self.out_channels = self.dataset.num_dist_bins

        kernel_size = (5, 5)
        self.model = nn.Sequential(
            ResNet2D(self.in_channels, block, num_blocks, init_channels=init_channels, kernel_size=5),
            nn.Conv2d(init_channels * pow(2, len(num_blocks) - 1), self.out_channels, kernel_size=kernel_size,
                      padding=(kernel_size[0]//2, kernel_size[1]//2)))

    def forward(self, x):
        return self.model(x)

    def train(self, **kwargs):
        train_and_validate(self.model, self.train_loader, self.test_loader,
                           class_weights=self.dataset.get_balanced_class_weights(indices=self.train_indices),
                           **kwargs)


class AntibodyGraphResNet(nn.Module):
    def __init__(self, h5file, num_blocks=10, batch_size=32, init_channels=64, num_workers=4):
        super(AntibodyGraphResNet, self).__init__()
        self.dataset = H5AntibodyDataset(h5file)
        indices = list(range(len(self.dataset)))
        self.train_indices, self.test_indices = indices[len(indices) // 10:], indices[:len(indices) // 10]
        self.train_loader = data.DataLoader(self.dataset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(self.train_indices),
                                            num_workers=num_workers)
        self.test_loader = data.DataLoader(self.dataset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(self.test_indices),
                                           num_workers=num_workers)
        self.init_channels = init_channels
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]

        feature, label = self.dataset[0]
        self.in_channels = feature.shape[0]
        self.out_channels = self.dataset.num_dist_bins
        self.graph_size = label.shape
        print(self.graph_size)

        kernel_size = (5, 5)
        self.model = nn.Sequential(
            Edge2EdgeResNet(self.in_channels, self.graph_size, num_blocks, init_channels=init_channels),
            nn.Conv2d(init_channels * pow(2, len(num_blocks) - 1), self.out_channels, kernel_size=kernel_size,
                      padding=(kernel_size[0]//2, kernel_size[1]//2)))

    def forward(self, x):
        return self.model(x)

    def train(self, **kwargs):
        train_and_validate(self.model, self.train_loader, self.test_loader,
                           class_weights=self.dataset.get_balanced_class_weights(indices=self.train_indices),
                           **kwargs)


if __name__ == '__main__':
    def main():
        from datetime import datetime
        h5file = '../data/ab_pdbs.h5'
        #resnet = AntibodyResNet(h5file, num_blocks=30)
        resnet = AntibodyGraphResNet(h5file, num_blocks=10)
        save_file = '{}_50blocks_{}.p'.format(resnet.model.__class__.__name__, datetime.now().strftime('%d-%m-%y_%H:%M:%S'))
        resnet.train(save_file='/scratch/cguerra5/' + save_file, epochs=10)
    main()

