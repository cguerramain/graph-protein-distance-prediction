import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D  # Not explicitly used, but needed for matplotlib 3d projections
from pylatexenc.latexencode import utf8tolatex


def _get_outfilename(inputfile):
    extracted_filename = (inputfile.split('/')[-1]).split('.')[0]
    return extracted_filename


def plot_lines(y_vectors, x=None, labels=None, title='Loss', xlabel='',
               ylabel='', caption='', out_file=None):
    """Plots a line graph for each y vector in a list of y vectors"""
    # Ensure all y_vectors are the same length
    lens = np.array([len(_) for _ in y_vectors])
    if not np.all(len(y_vectors[0]) == lens):
        raise ValueError('All vectors in y_vectors must be the same length')

    # Default x and labels to array indices
    if x is None:
        x = range(len(y_vectors[0]))
    if labels is None:
        labels = [str(_) for _ in range(len(y_vectors[0]))]

    rc('text', usetex=True)
    full_xlabel = _add_caption(xlabel, caption)

    for y, label in zip(y_vectors, labels):
        plt.plot(x, y, label=utf8tolatex(label))

    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel(full_xlabel)
    plt.ylabel(ylabel)

    # Output to file or to screen
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    rc('text', usetex=False)

    plt.close()


def heatmap2d(matrix, title='Heatmap', ylabel='', xlabel='', caption='',
              color_min=None, color_max=None, out_file=None, line_indices=None,
              line_color='r', line_color_other='k', xticks=None, yticks=None):
    """Displays the heatmap of a 2D matrix

    :param matrix: The matrix to turn into a heat map.
    :type matrix: numpy.ndarray
    :param title: Title of the heat map.
    :type title: str
    :param caption: The caption under the x-axis
    :type caption: str
    :param color_min: The minimum value on the color bar.
    :param color_max: The maximum value on the color bar.
    :param out_file:
        The file to output to. If None, the heatmap is output to the screen.
    :param line_indices:
        A list of indices to add vertical and horizontal lines to.
    :param line_color: The color of the lines specified by line_indices.
    :param y_max: Max y value
    :param y_min: min y value
    :param x_max: Max x value
    :param x_min: Min x value
    :rtype: None
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if line_indices is None:
        line_indices = {}

    full_xlabel = _add_caption(xlabel, caption)

    #rc('text', usetex=True)
    plt.imshow(matrix, cmap='viridis')

    ax = plt.gca()

    if xticks:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        print(xticks)
    if yticks:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)
        print(yticks)

    # Add color limits

    plt.colorbar()
    plt.clim(color_min, color_max)
    plt.title(title)
    plt.xlabel(full_xlabel)
    plt.ylabel(ylabel)

    # Explicitly set x and y limits (adding lines will extend the limits if this
    # is not done)
    plt.ylim(( len(matrix), 0))
    plt.xlim((0, len(matrix[0])))

    # Add horizontal and vertical lines
    for key in line_indices:
        list_indices=line_indices[key]
        if key=='h3':
            for idx in list_indices:
                plt.vlines(idx - 0.5, ymin=0, ymax=len(matrix[0]), color=line_color)
                plt.hlines(idx - 0.5, xmin=0, xmax=len(matrix), color=line_color)
        else:
            for idx in list_indices:
                plt.vlines(idx - 0.5, ymin=0, ymax=len(matrix[0]), color=line_color_other)
                plt.hlines(idx - 0.5, xmin=0, xmax=len(matrix), color=line_color_other)

    # Output to file or to screen
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    # rc('text', usetex=False)

    plt.close()


def binned_matrix(in_tensor, method='max'):
    """
    Bins a 3d tensor, in_tensor, of shape (channels, N, N) to an integer tensor,
    out_tensor, of shape (N, N) where each element out_tensor[i][j] equals the
    index of the maximum value or the average value of in_tensor[:, i, j].
    :param in_tensor: The tensor to bin.
    :type in_tensor: torch.Tensor
    :param method:
        The binning method. Can either be 'max' or 'average'. 'max' will
        assign an element to the bin with the highest probability and 'average'
        will assign an element to the weighted average of the bins
    :return:
    """
    print(in_tensor[0])
    in_tensor = nn.Softmax(dim=2)(torch.einsum('cij -> ijc', in_tensor))
    if method == 'max':
        # Predict the bins with the highest probability
        return in_tensor.max(2)[1]
    elif method == 'avg':
        # Predict the bin that is closest to the average of the probability dist
        # predicted_bins[i][j] = round(sum(bin_index * P(bin_index at i,j)))
        bin_indices = torch.arange(in_tensor.shape[-1]).float()
        predicted_bins = torch.round(torch.sum(in_tensor.mul(bin_indices),
                                               dim=len(in_tensor.shape)-1))
        return predicted_bins
    else:
        raise ValueError('method must be in {\'avg\',\'max\'}')


def plot_model_output(model, input_, mask=None, mask_fill_value=-1, method='max', **kwargs):
    output = binned_matrix(model(input_)[0], method=method)
    if mask is not None:
        output[mask] = mask_fill_value
    heatmap2d(output, **kwargs)


def _add_caption(label, caption):
    """Adds a caption to a label using LaTeX"""
    full_label = ''
    if label != '' or caption != '':
        if label != '':
            label = utf8tolatex(label)
            label = label + r'\\*'
        if caption != '':
            caption = utf8tolatex(caption)
            caption = r'\textit{\small{' + caption + r'}}'
        full_label = r'\begin{center}' + label + caption + r'\end{center}'
    return full_label


if __name__ == '__main__':
    def main():
        from src.models import AntibodyResNet, AntibodyGraphResNet
        saved_model = '../saved_models/Antibody_GCNN_epochs30_lr0p0001_batch_size64_num_blocks10_epoch30.p'
        h5file = '../data/ab_pdbs.h5'
        resnet = AntibodyGraphResNet(h5file, num_blocks=10)
        model = resnet.model
        model.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu')))
        for i in range(len(resnet.dataset)):
            feature, label = resnet.dataset[i]
            mask = resnet.dataset.get_mask(i)
            feature = feature.unsqueeze(0)
            plot_model_output(model, feature, mask=mask, color_min=0, color_max=32)
            heatmap2d(label, title='Actual Distance Matrix', color_min=0, color_max=32)
    main()

