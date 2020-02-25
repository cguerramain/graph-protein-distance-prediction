import h5py
import warnings
import numpy as np
import src.preprocess.pdb_parser as pdb_parser
import argparse
from tqdm import tqdm
from os import listdir, remove
from os.path import join, isfile
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


def antibody_to_h5(pdb_dir, out_file_path, offset=8,
                   overwrite=False, print_progress=False,
                   slice_size=64):
    if overwrite and isfile(out_file_path):
        remove(out_file_path)
    
    h5_out = create_datasets(out_file_path, slice_size)
    pdb_files = [join(pdb_dir, _) for _ in listdir(pdb_dir) if _[-3:] == 'pdb']
    idx = 0
    for file in tqdm(pdb_files, disable=(not print_progress), total=len(pdb_files)):
        idx = add_pdb_to_dataset(h5_out, file, idx, offset=offset, slice_size=slice_size)
    h5_out.close()


def add_pdb_to_dataset(h5_file, pdb_file, idx, offset=0, slice_size=64):
    """
    :param h5_file:
    :param pdb_file:
    :param idx:
    :param offset:
    :param slice_size:
    :return:
    """
    pdb_id = pdb_parser.filename_no_extension(pdb_file)

    seqs = pdb_parser.aa_seq(pdb_file)
    if seqs is None:
        seqs = pdb_parser.aa_seq_from_coords(pdb_file)
    terts = pdb_parser.CB_coords(pdb_file)
    if not terts:
        warnings.warn('WARNING: Skipping {}'.format(pdb_file))
        return idx
    masks = pdb_parser.mask_aa_coords(pdb_file)
    for chain_id in seqs.keys():
        seq = np.array(pdb_parser.aa_seq_to_num(seqs[chain_id]), dtype=np.uint8)
        seq_len = len(seq)
        indices = np.array(range(1, len(seq) + 1), dtype=np.int16)
        tert = np.array(terts[chain_id], dtype=float)
        mask = np.array(masks[chain_id], dtype=np.uint8)

        # Add appropriate padding
        seq = np.pad(seq, offset, 'constant')
        indices = np.pad(indices, offset, 'constant')
        tert = np.pad(tert, ((offset, offset), (0, 0)), 'constant')
        mask = np.pad(mask, offset, 'constant')

        for i in range(0, len(seq), slice_size):
            # Resize to fit the new data element
            if idx >= h5_file['pdb_id'].shape[0]:
                for key in h5_file.keys():
                    # Add row to dataset
                    dataset = h5_file[key]
                    resized_shape = list(dataset.shape)
                    resized_shape[0] += 1
                    dataset.resize(resized_shape)

            h5_file['pdb_id'][idx] = np.string_(pdb_id)
            h5_file['chain_id'][idx] = np.string_(chain_id)
            h5_file['sequence_length'][idx] = seq_len
            h5_file['primary'][idx, :len(seq[i:i+slice_size])] = seq[i:i+slice_size]
            h5_file['indices'][idx, :len(indices[i:i+slice_size])] = indices[i:i+slice_size]
            h5_file['tertiary'][idx, :len(tert[i:i+slice_size])] = tert[i:i+slice_size]
            h5_file['mask'][idx, :len(mask[i:i+slice_size])] = mask[i:i+slice_size]

            idx += 1
    return idx


def create_datasets(out_file_path, slice_size):
    """"""
    h5_out = h5py.File(out_file_path, 'w')
    pdb_id_set = h5_out.create_dataset('pdb_id', (1,),
                                       compression='lzf', dtype='S25',
                                       maxshape=(None,))
    
    chain_id_set = h5_out.create_dataset('chain_id', (1,),
                                         compression='lzf', dtype='S1',
                                         maxshape=(None,))

    indices_set = h5_out.create_dataset('indices',
                                        (1, slice_size),
                                        compression='lzf', dtype='int16',
                                        maxshape=(None, slice_size),
                                        fillvalue=0)

    prim_set = h5_out.create_dataset('primary',
                                     (1, slice_size),
                                     compression='lzf', dtype='uint8',
                                     maxshape=(None, slice_size),
                                     fillvalue=0)
    
    tert_set = h5_out.create_dataset('tertiary',
                                     (1, slice_size, 3),
                                     compression='lzf', dtype='float',
                                     maxshape=(None, slice_size, 3),
                                     fillvalue=0)

    mask_set = h5_out.create_dataset('mask',
                                     (1, slice_size),
                                     compression='lzf', dtype='uint8',
                                     maxshape=(None, slice_size),
                                     fillvalue=0)

    seq_len = h5_out.create_dataset('sequence_length',
                                    (1,),
                                    maxshape=(None,), dtype='int16')

    return h5_out


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_dir_path', type=str,
                        help=('The path to the directory with preprocessed '
                              'antibody PDB files'))
    parser.add_argument('out_file_path', type=str,
                        help='The output path to save the H5 dataset to')
    parser.add_argument('--offset', type=int, default=8,
                        help=('The amount of zero padding to append on each '
                              'side of the protein sequence'))
    parser.add_argument('--slice_size', type=int, default=64,
                        help=('The size of each slice of a protein sequence. '
                              'For instance, a slice_size of 64 would slice '
                              'a sequence of length 192 into 3 equal length '
                              'subsequences'))
    args = parser.parse_args()
    antibody_to_h5(pdb_dir=args.pdb_dir_path, out_file_path=args.out_file_path,
                   offset=args.offset, slice_size=args.slice_size, print_progress=True,
                   overwrite=True)


if __name__ == '__main__':
    _cli()

