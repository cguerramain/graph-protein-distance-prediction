import torch
import torch.nn as nn
import torch.sparse as sparse


seperation_ranges = {'short': [6, 11], 'medium': [12, 23],
                     'long': [24, float('inf')]}


def top_k_predictions_(logits, seq_lens, k=2, mask=None, ang8_bin=8):
    """
    :param logits: A tensor of shape (batch, logits, height, width)
    :param k:
    :param mask:
    :return:
    """
    pairwise_probs = pairwise_contact_probs(logits, mask=mask, ang8_bin=ang8_bin)
    return [probs[:seq_lens[i] // k] for i, probs in enumerate(pairwise_probs)]


def top_k_contact_metrics(features, logits, dist_mat, seq_lens, k=5, contact_range='long', sequence_position_index1=0,
                          sequence_position_index2=22, **kwargs):
    """Calculates metrics for the top L/k predicted contacts
    :param logits: The logits to generate probabilities from. Should have shape
                   (batch, logits, n, n).
    :type logits: torch.Tensor
    :param true_dist_mat:
    :param k:
    :param kwargs: kwargs to pas to top_k_contacts
    :return:
    """
    from viz import heatmap2d
    features = torch.einsum('bcij -> bijc', features)

    # Get mask from the non-positive distances of the distance matrix
    # (ignores 0 as well because those are not counted in the top-k metric according to CASP)
    mask = torch.zeros(dist_mat.shape, dtype=torch.uint8)
    mask[dist_mat > 0] = 1

    residue_distances = features[:, :, :, sequence_position_index1] - features[:, :, :, sequence_position_index2]
    print(residue_distances[0])
    residue_distances[~mask] = 0
    # Use only the top diagonal side of the distance matrix
    residue_distances[residue_distances <= 0] = 0

    lower_bound, upper_bound = seperation_ranges[contact_range]
    mask[(residue_distances < lower_bound).__or__(residue_distances > upper_bound)] = 0
    dist_mat[~mask] = -1
    heatmap2d(dist_mat[0])

    # True positive, False positive, False negative calculations
    tp = len(predicted_contact_mat[predicted_contact_mat.__and__(true_contact_mat)])
    fp = int(predicted_contacts.shape[0]) - tp
    fn = total_contacts - tp

    # Calculate metrics
    if tp + fp == 0:
        precision = float('nan')
    else:
        precision = tp / float(tp + fp)
    if tp + fn == 0:
        recall = float('nan')
    else:
        recall = tp / float(tp + fn)
    if precision + recall == 0:
        f1 = float('nan')
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return torch.Tensor([precision, recall, f1])


def binned_dist_mat_contact_metrics(logits, true_binned_dist_mat, ang8_bin=8, **kwargs):
    """Gets the contact metrics of logits and a binned distance matrix label
    :param logits: The logits to generate probabilities from. Should have shape
                   (logits, n, n).
    :type logits: torch.Tensor
    :param ang8_bin:
        The index of the bin containing distances between 7.5 and 8 angstroms.
        It is assumed that every index prior is <8 angstroms
    :type ang8_bin: int
    """
    # Turn the distance matrix into a contact matrix
    dist_mat = torch.Tensor(true_binned_dist_mat.shape).fill_(ang8_bin + 8)
    dist_mat[(true_binned_dist_mat <= ang8_bin).__and__(true_binned_dist_mat >= 0)] = 1

    return top_k_contact_metrics(logits, dist_mat, ang8_bin=ang8_bin, **kwargs)


def batch_binned_dist_mat_contact_metrics(logits, true_binned_dist_mats, ang8_bin=8, **kwargs):
    """Gets the average of each contact metric for a batch of binned distance matrices
    :param logits: The logits to generate probabilities from. Should have shape
                   (batch_size, logits, n, n).
    :type logits: torch.Tensor
    :param ang8_bin:
        The index of the bin containing distances between 7.5 and 8 angstroms.
        It is assumed that every index prior is <8 angstroms
    :type ang8_bin: int
    """
    metrics = torch.zeros(3)
    for i in range(logits.shape[0]):
        metrics += binned_dist_mat_contact_metrics(
            logits[i], true_binned_dist_mats[i], ang8_bin=ang8_bin, **kwargs)
    return metrics / logits.shape[0]


def contact_probs(logit_batch, ang8_bin=8):
    """"""
    if len(logit_batch.shape) != 4:
        raise ValueError('Expected a shape with three dimensions (batch, logits, L, L), got {}'.format(logit_batch.shape))
    logits_last = torch.einsum('bcij -> bijc', logit_batch)
    probs = nn.Softmax(dim=3)(logits_last)
    # Sum up the probability that any given residue pair is in contact (<8 Ang.)
    return probs[:, :, :, :ang8_bin+1].sum(3)


def mask_matrix_indices_(matrix_2d, mask_1d, not_mask_fill_value=-1):
    """
    Given a mask which is a 1D tensor of 1's and 0's and a 2D square tensor of the same length as the mask,
    this function

    :param matrix_2d:
    :param mask_1d:
    :param not_mask_fill_value:
    :return:
    """
    n = len(mask_1d)
    mask_1d = mask_1d.unsqueeze(0)  # Expand to two dimensions
    mask_1d = mask_1d.expand(n, n) + mask_1d.transpose(0, 1)
    matrix_2d[mask_1d > 0] = not_mask_fill_value
    return matrix_2d


def pairwise_contact_probs(logits, mask=None, ang8_bin=8):
    probs = contact_probs(logits, ang8_bin=ang8_bin)
    if mask is not None:
        if not isinstance(mask, torch.ByteTensor):
            raise ValueError('ERROR: Expected a ByteTensor for mask, got {}'.format(type(mask)))
        probs[~mask] = -1

    batch_size, height, width = probs.shape
    indexed_probs = torch.zeros((batch_size, height, width, 3))
    indexed_probs[:, :, :, 0] = torch.arange(0, height).unsqueeze(0).transpose(0, 1)
    indexed_probs[:, :, :, 1] = torch.arange(0, width)
    indexed_probs[:, :, :, 2] = probs

    def sort_batch(batch):
        return torch.Tensor(sorted(batch[batch[:, 2] > 0].tolist(), key=lambda x: -x[2]))
    return [sort_batch(batch) for batch in indexed_probs.reshape((batch_size, -1, 3))]


if __name__ == '__main__':
    def main():
        import torch.nn.functional as F
        from viz import heatmap2d
        from data import H5AntibodyDataset

        dataset = H5AntibodyDataset('../data/ab_pdbs.h5')
        feature, label, index = dataset[0]
        label += 1
        label[label < 0] = 0
        dist_mat = label.clone().unsqueeze(0)
        [print(_) for _ in dist_mat[0]]
        label = F.one_hot(dist_mat, num_classes=33)
        print(label.shape)
        heatmap2d(dist_mat[0])
        seq_len = torch.Tensor([dataset.get_sequence_length(index)]).unsqueeze(0)
        top_k_contact_metrics(feature.unsqueeze(0), label, dist_mat, seq_lens=seq_len, contact_range='short')
        '''
        logits = torch.rand((4, 3, 5, 20))
        mask = torch.ones((4, 3, 5)).byte()
        mask[:, :, 0] = 0
        logits = torch.einsum('bijc -> bcij', logits)
        print(logits)
        print(pairwise_contact_probs(logits, mask=mask))
        '''
    main()

'''
def top_k_predictions(logits, k=2, contact_range='long', residue_ranges=None,
                      **kwargs):
    """
    Outputs the highest probability L/k predictions L is the length of the
    amino acid sequence.
    :param k:
    :param logits: The logits to generate probabilities from. Should have shape
                   (logits, n, n).
    :type logits: torch.Tensor
    :param kwargs:
    :return:
    """
    if residue_ranges is not None:
        if isinstance(residue_ranges[0], int):
            residue_ranges = [residue_ranges]
        seq_len = sum([ub - lb + 1 for lb, ub in residue_ranges])
    else:
        seq_len = logits.shape[1]

    probs = torch.Tensor(pairwise_contact_probs(logits, **kwargs))

    if contact_range != 'all':
        if contact_range not in seperation_ranges:
            msg = '{} is not a valid contact_range. The range must be in ' \
                  '{\'short\', \'medium\', \'long\', \'all\'}'
            raise ValueError(msg.format(contact_range))

        if residue_ranges is not None:
            for lb, ub in residue_ranges:
                i_mask = (probs[:, 0] >= lb).__and__(probs[:, 0] <= ub)
                j_mask = (probs[:, 1] >= lb).__and__(probs[:, 1] <= ub)
                mask = i_mask.__or__(j_mask)
                probs = probs[mask]

        # Filter down to residues within a given range
        lower_bound, upper_bound = seperation_ranges[contact_range]
        seperations = probs[:, 1] - probs[:, 0] - 1
        mask = (seperations >= lower_bound).__and__(seperations <= upper_bound)
        probs = probs[mask]

    if k is not None:
        top_k = probs[:seq_len // k]
        return top_k
    else:
        return probs
'''

