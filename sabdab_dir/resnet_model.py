import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from layers.ResNet2D import ResNet2D, ResBlock2D
from H5AntibodyDataset import h5_antibody_dataloader, H5AntibodyDataset
from DeviceDataLoader import DeviceDataLoader
from torchsummary import summary
from viz import heatmap2d


class AntibodyResNet:
    def __init__(self, h5file, num_blocks=10, batch_size=32, init_channels=64):
        self.dataset = H5AntibodyDataset(h5file)
        indices = list(range(len(self.dataset)))
        self.train_indices, self.test_indices = indices[len(indices) // 10:], indices[:len(indices) // 10]
        self.train_loader = DeviceDataLoader(data.DataLoader(self.dataset, batch_size=batch_size,
                                                             sampler=SubsetRandomSampler(self.train_indices)))
        self.test_loader = DeviceDataLoader(data.DataLoader(self.dataset, batch_size=batch_size,
                                                            sampler=SubsetRandomSampler(self.test_indices)))
        
        self.init_channels = init_channels
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        
        feature, _ = self.dataset[0]
        self.in_channels = feature.shape[0]
        self.out_channels = self.dataset.num_dist_bins
        
        kernel_size = (5, 5)
        self.model = nn.Sequential(
            ResNet2D(self.in_channels, ResBlock2D, num_blocks, init_channels=init_channels, kernel_size=5),
            nn.Conv2d(init_channels * pow(2, len(num_blocks) - 1), self.out_channels, kernel_size=kernel_size, 
                      padding=(kernel_size[0]//2, kernel_size[1]//2)))

    def train(self, max_iter=float('Inf'), epochs=4, lr=1e-5):
        self.model.train()
        total_batches = (len(self.train_indices) // self.train_loader.dl.batch_size) + 1
        total_iters = min(max_iter, total_batches)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.train_loader.device, non_blocking=True)
        print('Number of parameters: {}'.format(count_parameters(self.model)))
        summary(self.model, input_size=(self.in_channels, 64, 64))
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (features, labels) in tqdm(enumerate(self.train_loader), total=total_iters):
                if i >= max_iter:
                    break

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                def handle_batch():
                    """Function done to ensure variables immediately get dealloced"""
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    return outputs, float(loss.item())

                outputs, batch_loss = handle_batch()
                running_loss += batch_loss
                print(batch_loss)
                # print(running_loss)


def count_parameters(model):
    """Counts the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    def main():
        AntibodyResNet('ab_pdbs.h5', num_blocks=[10, 10, 5]).train()
    main()

