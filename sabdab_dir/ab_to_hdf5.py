import h5py
import warnings
import numpy as np
import pdb_parser as parser
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
    pdb_id = parser.filename_no_extension(pdb_file)

    seqs = parser.aa_seq(pdb_file)
    terts = parser.CB_coords(pdb_file)
    if not terts:
        warnings.warn('WARNING: Skipping {}'.format(pdb_file))
        return idx
    masks = parser.mask_aa_coords(pdb_file)
    for chain_id in seqs.keys():
        seq = np.array(parser.aa_seq_to_num(seqs[chain_id]), dtype=np.uint8)
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
            h5_file['indices'][idx, :len(indices[i:i+slice_size])] = indices[i:i+slice_size]
            h5_file['primary'][idx, :len(seq[i:i+slice_size])] = seq[i:i+slice_size]
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
    
    return h5_out


if __name__ == '__main__':
    def main():
        pdb_dir = 'pdbs/'
        out_file_path = 'ab_pdbs.h5'
        antibody_to_h5(pdb_dir, out_file_path, offset=8, print_progress=True)
    main()

