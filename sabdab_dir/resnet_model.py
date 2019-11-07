import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from layers.ResNet2D import ResNet2D, ResBlock2D
from H5AntibodyDataset import h5_antibody_dataloader, H5AntibodyDataset
from DeviceDataLoader import DeviceDataLoader
from viz import heatmap2d


class AntibodyResNet:
    def __init__(self, h5file, num_blocks=10, batch_size=32):
        self.dataset = H5AntibodyDataset(h5file)
        indices = list(range(len(self.dataset)))
        self.train_indices, self.test_indices = indices[len(indices) // 10:], indices[:len(indices) // 10]
        self.train_loader = DeviceDataLoader(data.DataLoader(self.dataset, batch_size=batch_size,
                                                             sampler=SubsetRandomSampler(self.train_indices)))
        self.test_loader = DeviceDataLoader(data.DataLoader(self.dataset, batch_size=batch_size,
                                                            sampler=SubsetRandomSampler(self.test_indices)))
        feature, label = self.dataset[0]
        self.in_planes = feature.shape[0]
        self.out_planes = label.shape[0]
        kernel_size = (5, 5)
        self.model = nn.Sequential(
            ResNet2D(self.in_planes, ResBlock2D, [num_blocks], init_planes=64, kernel_size=5),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)))

    def train(self, max_iter=None):
        self.model.train()
        total_batches = max_iter
        if max_iter is None:
            total_batches = (len(self.train_indices) // self.train_loader.batch_size) + 1
            max_iter = float('Inf')

        running_loss = 0.0
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(self.model.parameters())
        for i, (features, labels) in tqdm(enumerate(self.train_loader), total=total_batches):
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


if __name__ == '__main__':
    def main():
        AntibodyResNet('ab_pdbs.h5', num_blocks=10).train()
    main()

