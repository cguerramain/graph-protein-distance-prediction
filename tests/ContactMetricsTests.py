import unittest
import torch
import torch.nn as nn
from src.contact_metrics import *
from src.viz import heatmap2d


class ContactMetricsTests(unittest.TestCase):
    def test_top_k_predictions(self):
        logits = torch.ones((2, 5, 5, 3))
        logits[0, :, 1, :] = 5
        logits[0, :, :, 1] = 2
        logits[0, :, :, 2] = 3
        logits = torch.einsum('bijc -> bcij', logits)
        heatmap2d(contact_probs(logits)[0])


if __name__ == '__main__':
    def main():
        unittest.main()
    main()
