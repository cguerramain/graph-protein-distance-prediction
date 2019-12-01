import math
import torch.nn as nn
import torch.nn.functional as F


class Edge2Edge(nn.Module):
    def __init__(self, in_channels, out_channels, graph_size):
        super(Edge2Edge, self).__init__()
        if len(graph_size) != 2:
            raise ValueError('ERROR: Expected a size of (height, width), got: {}'.format(graph_size))

        height, width = self.graph_size = graph_size
        self.vertical_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(height, 1))
        self.horizontal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, width))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        vert_conv = self.vertical_conv(x)
        hort_conv = self.horizontal_conv(x)
        return vert_conv.expand((batch_size, -1, height, width)).add(hort_conv)


class Edge2EdgeResBlock(nn.Module):
    """A basic residual block with 2D CNNs
    Defines a convolutional ResNet block with the following architecture:
    -- Shortcut Path -->
    +-------- Shortcut Layer --------+
    |                                |
    X ->  e2e  ->  Act  ->  e2e  -> Sum -> Act -> Output
    -- Main Path -->
    The shortcut layer defaults to zero padding if the non-plane dimensions of
    X do not change after convolutions. Otherwise, it defaults to a 2D
    convolution to match dimensions.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, graph_size, shortcut=None,
                 activation=F.relu):
        """
        :param in_channels: The number of input channels (features)
        :param out_channels: The number of output channels (features)
        :param kernel_size: Width of the kernel used in the Conv2D convolution.
        :param stride: Stride used in the Conv2D convolution.
        :param shortcut: Callable function to for the shortcut path
        """
        super(Edge2EdgeResBlock, self).__init__()
        self.activation = activation

        self.e2e1 = Edge2Edge(in_channels, out_channels, graph_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.e2e2 = Edge2Edge(out_channels, out_channels, graph_size)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Default zero padding shortcut
        if shortcut is None:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, 0, 0, out_channels - x.shape[1], 0, 0))
        # User defined shortcut
        else:
            self.shortcut = shortcut

    def forward(self, x):
        """
        :param x: A FloatTensor to propagate forward
        :type x: torch.Tensor
        :return:
        """
        out = self.activation(self.bn1(self.e2e1(x)))
        out = self.bn2(self.e2e2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Edge2EdgeResNet(nn.Module):
    def __init__(self, in_channels, graph_size, num_blocks, init_channels=64,
                 activation=F.relu):
        super(Edge2EdgeResNet, self).__init__()
        if not (init_channels != 0 and ((init_channels & (init_channels - 1)) == 0)):
            raise ValueError('The initial number of planes must be a power of 2')

        self.activation = activation
        self.init_planes = init_channels
        self.in_planes = self.init_planes  # Number of input planes to the final layer
        self.num_layers = len(num_blocks)
        self.graph_size = graph_size

        self.e2e1 = Edge2Edge(in_channels, self.init_planes, graph_size)
        self.bn1 = nn.BatchNorm2d(self.init_planes)

        self.layers = []
        # Raise the number of planes by a power of two for each layer
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(Edge2EdgeResBlock, int(self.init_planes * math.pow(2, i)), num_blocks[i])
            self.layers.append(new_layer)

            # Done to ensure layer information prints out when print() is called
            setattr(self, 'layer{}'.format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, self.graph_size))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.e2e1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == '__main__':
    import torch
    t = torch.Tensor(
        [
            [
                [[1], [1], [1]],
                [[2], [2], [2]],
                [[3], [3], [3]]
            ]
        ])
    t = torch.einsum('bijc -> bcij', t)
    e2e = Edge2Edge(1, 1, (3, 3))
    e2e.vertical_conv.weight.data.fill_(3)
    e2e.vertical_conv.bias.data.fill_(0)
    e2e.horizontal_conv.weight.data.fill_(7)
    e2e.horizontal_conv.bias.data.fill_(0)
    out = e2e(t)
    print(torch.einsum('bcij -> bijc', out))

