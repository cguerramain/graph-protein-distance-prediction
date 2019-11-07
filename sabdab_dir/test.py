import torch
import torch.nn as nn
import torch.nn.functional as F


in_ = torch.LongTensor([
    [
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
    ],
    [
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
    ]])

in_ = F.one_hot(in_).float()
in_ = torch.einsum('bijc -> bcij', in_)

label = torch.LongTensor([
    [
        [0, 1, 2, 4, 3],
        [0, 1, 2, -1, 5],
    ],
    [
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
    ]])

print(nn.CrossEntropyLoss(ignore_index=-1)(in_, label))

