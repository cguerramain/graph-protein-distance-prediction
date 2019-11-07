import torch
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
              line_color='r',line_color_other='k',xticks=None,yticks=None):
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

    if not xticks is None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        print(xticks)
    if not yticks is None:
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
    #rc('text', usetex=False)

    plt.close()


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

