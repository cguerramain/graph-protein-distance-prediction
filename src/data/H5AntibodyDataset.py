import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm


class H5AntibodyDataset(data.Dataset):
    def __init__(self, file_name, num_dist_bins=32):
        """
        :param file_name: The h5 file for the antibody data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        :param num_dist_bins:
            The number of bins to discretize the distance matrix into. If None,
            then the distance matrix remains continuous.
        """
        super(H5AntibodyDataset, self).__init__()

        self.h5file_name = file_name
        self.num_dist_bins = num_dist_bins

        self.bins = get_bins(self.num_dist_bins)
        self.squared_bins = [(pow(l, 2), pow(r, 2)) for l, r in self.bins]
        self.chain_ranges = self._get_chain_ranges()
        self.indices = self._get_indices()

    def __getitem__(self, index):
        """Gets the combined sequence information and distance matrix at an index"""
        if isinstance(index, slice):
            pairs = []
            start, stop, step = index.indices(len(self))
            for i in range(start, stop, step):
                distance_matrix = self.get_distance_matrix(i)
                sequence_info = self.get_positional_sequence_information(i)
                pairs.append([sequence_info, distance_matrix])
            return pairs
        else:
            distance_matrix = self.get_distance_matrix(index)
            sequence_info = self.get_positional_sequence_information(index)
            return sequence_info, distance_matrix

    def __len__(self):
        return len(self.indices)

    def get_pdb_id(self, index):
        h5index1, _ = self.indices[index]
        return self.get_h5file()['pdb_id'][h5index1]

    def get_chain_id(self, index):
        h5index1, _ = self.indices[index]
        return self.get_h5file()['chain_id'][h5index1]

    def get_mask(self, index):
        h5index1, h5index2 = self.indices[index]
        return self._combine_masks(h5index1, h5index2)

    def get_positional_sequence_information(self, index):
        h5index1, h5index2 = self.indices[index]
        return self._combine_sequences(h5index1, h5index2)

    def get_h5file(self):
        """Returns a h5py File instance of the dataset - needed for handling multiple threads"""
        return h5py.File(self.h5file_name, 'r', swmr=True)

    def get_distance_matrix(self, index, bin_matrix=True, mask_fill_value=-1):
        h5index1, h5index2 = self.indices[index]
        h5file = self.get_h5file()
        h1coords = torch.FloatTensor(h5file['tertiary'][h5index1])
        h2coords = torch.FloatTensor(h5file['tertiary'][h5index2])

        h1coords = h1coords.unsqueeze(0)
        dist_mat_shape = (h1coords.shape[1], h1coords.shape[1], h1coords.shape[2])
        row_expand = h1coords.transpose(0, 1).expand(dist_mat_shape)

        if self.num_dist_bins != 0 and bin_matrix:
            # If the exact distance isn't required, then save time on
            # calculations by calculating norm without the square root and
            # squaring all the bin values, yields same bins.
            dist_mat = row_expand.add(-h2coords).pow(2).sum(2)
            dist_mat = bin_distance_matrix(dist_mat, bins=self.squared_bins)
        else:
            dist_mat = (row_expand.add(-h2coords)).norm(dim=2)
        dist_mat[self.get_mask(index)] = mask_fill_value
        return dist_mat

    def get_all(self, index):
        h5index1, h5index2 = self.indices[index]
        h5file = self.get_h5file()
        return dict(
            pdb_id=self.get_pdb_id(index),
            chain_id=self.get_chain_id(index),
            sequence1=h5file['primary'][h5index1],
            sequence2=h5file['primary'][h5index2],
            indices1=h5file['indices'][h5index1],
            indices2=h5file['indices'][h5index2],
            coords1=h5file['tertiary'][h5index1],
            coords2=h5file['tertiary'][h5index2],
            combined_sequence=self.get_positional_sequence_information(index),
            distance_matrix=self.get_distance_matrix(index))

    def get_balanced_class_weights(self, indices=None, show_progress=True):
        """Estimates the weights for unbalanced classes in a onehot encoded dataset
        Uses the following equation to estimate weights:
        ``n_samples / (num_bins * torch.bincount(bin))``

        :param sample_percentage:
            The percentage of the data used when calculating the class weights
        """
        if self.num_dist_bins < 0:
            raise ValueError('The number of bins must be greater than or equal to 1')
        if not indices:
            indices = range(len(self.indices))

        if show_progress:
            print('Calculating class weights...')
        bin_counts = torch.zeros([self.num_dist_bins], dtype=torch.long)
        for idx in tqdm(indices, disable=(not show_progress)):
            binned_distance_matrix = self.get_distance_matrix(idx)
            for bins in binned_distance_matrix:
                # Ignore bins that are -1
                bin_count = torch.bincount(bins.int() + 1)[1:]
                bin_counts[:bin_count.size(0)] += bin_count

        denominator = bin_counts * self.num_dist_bins
        weights = sum(bin_counts) / denominator.float()
        # If a class was not in the dataset, weigh it as much as the highest
        # weighted class
        weights[denominator == 0] = -1  # Get rid of inf values
        weights[denominator == 0] = max(weights)
        if show_progress:
            print('Bin weights: {}'.format(weights))
        return weights

    def _get_chain_ranges(self):
        h5file = self.get_h5file()
        num_slices = len(h5file['pdb_id'])
        if num_slices == 0:
            return []
        if num_slices == 1:
            return [(0, 0)]

        ranges = []
        left = 0
        pdb_ids, chain_ids = h5file['pdb_id'], h5file['chain_id']
        for i in range(1, num_slices):
            prev_chain_id, chain_id = chain_ids[i - 1], chain_ids[i]
            prev_pdb_id, pdb_id = pdb_ids[i - 1], pdb_ids[i]

            # If previous chain or pdb is different than the current, then a
            # new chain was hit. Add the previous chain's range to the list
            if prev_chain_id != chain_id or prev_pdb_id != pdb_id:
                ranges.append((left, i - 1))
                left = i
        ranges.append((left, num_slices - 1))  # Add last chain
        return ranges

    def _get_indices(self):
        if len(self.chain_ranges) == 0:
            return []

        indices = []
        for chain_start, chain_end in self.chain_ranges:
            for slice_start in range(chain_start, chain_end + 1):
                for slice_end in range(chain_start, chain_end + 1):
                    indices.append((slice_start, slice_end))
        return indices

    def _combine_masks(self, h5index1, h5index2):
        """
        :param h5index1:
        :param h5index2:
        :return:
        """
        h5file = self.get_h5file()
        mask1, mask2 = torch.Tensor(h5file['mask'][h5index1]), torch.Tensor(h5file['mask'][h5index2])
        # Set 1's to 0's and vice versa
        not_mask1 = torch.ones(len(mask1)).type(dtype=mask1.dtype) - mask1
        not_mask2 = torch.ones(len(mask2)).type(dtype=mask2.dtype) - mask2

        # Applies not_mask1 to every row and not_mask2 to every column
        # Specifically: Expands not_mask1 to an nxn Tensor such that row i is
        # filled with not_mask1[i]'s value, then add not_mask2 to each row.
        # Example:
        # not_mask1 = [0, 0, 1, 1, 0], not_mask2 = [0, 1, 0, 0, 1]
        #             |0, 0, 0, 0, 0|   |0, 1, 0, 0, 1|   |0, 1, 0, 0, 1|
        #             |0, 0, 0, 0, 0|   |0, 1, 0, 0, 1|   |0, 1, 0, 0, 1|
        # operation = |1, 1, 1, 1, 1| + |0, 1, 0, 0, 1| = |1, 2, 1, 1, 2|
        #             |1, 1, 1, 1, 1|   |0, 1, 0, 0, 1|   |1, 2, 1, 1, 2|
        #             |0, 0, 0, 0, 0|   |0, 1, 0, 0, 1|   |0, 1, 0, 0, 1|
        n = len(not_mask1)
        not_mask = not_mask1.unsqueeze(1).expand(n, n).add(not_mask2)
        return not_mask > 0

    def _combine_sequences(self, h5index1, h5index2, onehot=True):
        """
        :param h5index1:
        :param h5index2:
        :return:
        """
        h5file = self.get_h5file()
        seq1, seq2 = torch.LongTensor(h5file['primary'][h5index1]), torch.LongTensor(h5file['primary'][h5index2])
        indices1, indices2 = torch.LongTensor(h5file['indices'][h5index1]), torch.LongTensor(h5file['indices'][h5index2])

        n = len(seq1)
        seq_row_expand = seq1.unsqueeze(1).expand(n, n)
        indices_row_expand = indices1.unsqueeze(1).expand(n, n).unsqueeze(2)
        seq_col_expand = seq2.unsqueeze(0).expand(n, n)
        indices_col_expand = indices2.unsqueeze(0).expand(n, n).unsqueeze(2)

        if onehot:
            num_amino_acids = 21
            seq_row_expand = F.one_hot(seq_row_expand, num_classes=num_amino_acids)
            seq_col_expand = F.one_hot(seq_col_expand, num_classes=num_amino_acids)
        else:
            seq_row_expand, seq_col_expand = seq_row_expand.unsqueeze(2), seq_col_expand.unsqueeze(2)
        combined = torch.cat((indices_row_expand, seq_row_expand, indices_col_expand, seq_col_expand), dim=2)
        # Switch shape from [timestep/length_i, timestep/length_j, filter/channel]
        #                to [filter/channel, timestep/length_i, timestep/length_j]
        combined = torch.einsum('ijc -> cij', combined)
        return combined.float()

    @staticmethod
    def merge_samples_to_minibatch(samples):
        features = torch.stack([f for f, _ in samples])
        labels = torch.stack([l for _, l in samples])
        return [features, labels]


def get_bins(num_bins, first_bin=4):
    if num_bins == 0:
        return []
    if num_bins == 1:
        return [(0, float('Inf'))]
    if num_bins == 2:
        return [(0, first_bin), (first_bin, float('Inf'))]
    bins = [(first_bin + 0.5 * i, first_bin + 0.5 * (1 + i)) for i in range(num_bins - 2)]
    bins.append((bins[-1][1], float('Inf')))
    bins.insert(0, (0, first_bin))
    return bins


def bin_distance_matrix(dist_matrix, bins=None):
    """Convert a continuous distance matrix to a binned version
    :param dist_matrix: A tensor of shape (n, n) of pairwise distances.
    :type dist_matrix: torch.Tensor
    :param bins:
        A list of two-tuples which define the bounds of a bin. By default, the
        bounds are: [(0, 4), (4, 4.5), (4.5, 5), ... (18.5, 19), (19, inf)]

        For instance, [(0, 4), (4, 4.5), (4.5, inf)] would be bounds
        for three bins. The first denoting distances in the range [0, 4), the
        second [4, 4.5), and the third [4.5, inf). Given these bins, the tensor
        [1, 4, 999.9, 0.1, -1] would be converted to [1, 2, 3, 1, 0] where the
        values correspond to the index of the bin they were placed in plus one.
        A value that does not belong to any bin is given 0 as a value.
    :type bins: torch.LongTensor()
    """
    if not bins:
        bins = get_bins(32)

    binned_matrix = -torch.ones(dist_matrix.shape, dtype=torch.long)
    # binned_matrix = torch.zeros(dist_matrix.shape, dtype=torch.long)
    for i, (lower_bound, upper_bound) in enumerate(bins):
        bin_mask = (dist_matrix >= lower_bound).__and__(dist_matrix < upper_bound)
        binned_matrix[bin_mask] = i
    return binned_matrix


def h5_antibody_dataloader(filename, batch_size=32, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError('Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(dict(collate_fn=H5AntibodyDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(H5AntibodyDataset(filename), **kwargs)


if __name__ == '__main__':
    def main():
        H5AntibodyDataset('../../data/ab_pdbs.h5')
    main()

