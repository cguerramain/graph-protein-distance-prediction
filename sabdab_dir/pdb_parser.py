import warnings
import numpy as np
import re
from os import listdir
from os.path import splitext, basename
from collections import defaultdict
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser


# Maps all amino acids to a number
aa_dict = {a: i + 1 for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY')}


def letter_to_num(string, dict_, unknown_value=-1):
    """ Convert string of letters to list of ints
    Code from ProteinNet (https://github.com/aqlaboratory/proteinnet)
    """
    return [dict_.get(s, unknown_value) for s in string]


def aa_seq_to_num(aa_seq):
    return letter_to_num(aa_seq, aa_dict, unknown_value=0)


def filename_no_extension(file_path):
    """Returns the name of a file without the extension"""
    return splitext(basename(file_path))[0]


def aa_seq(pdb_file):
    """
    Gets the full sequence of each protein chain from the SEQRES section of a 
    PDB file, if present.

    :return: 
        A dictionary mapping each protein chain to its full sequence from the
        pdb file, irrespective of whether or not a residue has a coordinate.
    :rtype: defaultdict(str)
    """
    # TODO: Try to get sequence from PDB API if the SEQRES section is missing 
    with open(pdb_file, 'r') as f:
        seq_lines = [l[:-1] for l in f.readlines() if l[:6] == 'SEQRES']

    seqs = defaultdict(str)
    for l in seq_lines:
        # Get the index of the last letter
        for i in reversed(range(len(l))):
            if l[i] != ' ':
                break
        chain_id = l[11]
        seq_3letters = l[19:i+1]
        seqs[chain_id] += seq1(''.join(seq_3letters.split(' ')))
    
    return seqs


def aa_seq_from_coords(pdb_file):
    """
    Gets the sequence of each protein chain from the ATOM section of a PDB 
    file. Any residue with a coordinate will be shown in the sequence, but any

    :return: 
        A dictionary mapping each protein chain to its sequence from the
        pdb file, for each residue that has at least one coordinate.
    :rtype: defaultdict(str)
    """
    p = PDBParser()
    file_name = filename_no_extension(pdb_file)
    structure = p.get_structure(file_name, pdb_file)
    
    seqs = {}
    for chain in structure.get_chains():
        seq3letters = ''.join([res.get_resname() for res in chain.get_residues()])
        seqs[chain.get_id()] = seq1(seq3letters, undef_code='')
    return seqs
        

def CB_coords(pdb_file, include_masks=False):
    """
    Gets the coordinates of the C-Beta atom in each residue or the C-Alpha atom
    if the residue does not have a C-beta coordinate. If a residue has neither,
    its coordinates are set to [0, 0, 0].
    
    An array mask can also be returned to denote non-existing coordinates, where
    mask[i] is denotes whether or not (1 or 0) the i-th residue has a
    coordinate.
    """
    p = PDBParser()
    file_name = filename_no_extension(pdb_file)
    structure = p.get_structure(file_name, pdb_file)
    
    def get_cb_or_ca(residue):
        if 'CB' in residue:
            return residue['CB'].get_coord()
        elif 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            return [0, 0, 0]
    
    coords = {}
    masks = mask_aa_coords(pdb_file)
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        if chain_id in masks:
            coords[chain_id] = np.zeros((len(masks[chain_id]), 3))
            chain_coords = [get_cb_or_ca(r) for r in chain.get_residues() if seq1(r.get_resname()) != 'X']
            if chain_coords:
                if len(chain_coords) != len(coords[chain_id][masks[chain_id]]):
                    msg = ('WARNING: In {}, chain {} the mask is not equal to '
                           'the number of coordinates. Returning None')
                    warnings.warn(msg.format(pdb_file, chain_id))
                    return None
                coords[chain_id][masks[chain_id]] = chain_coords
        else:
            msg = ('WARNING: Chain ID mismatch between the full sequences and the '
                   'sequences derived from coordinates in the {} file. Chain {} in '
                   'the full sequence is not in the sequences derived from ' 
                   'coordinates. Skipping chain {}')
            warnings.warn(msg.format(pdb_file, chain_id, chain_id))

    if include_masks:
        return coords, masks
    else:
        return coords


def align_strings(s1, s2):
    """
    Aligns a shorter string with a longer string. An example of an alignment a:
    s1 = ABBCTGCQQHN
     a = AB-CTGC--H-
    s2 = AB CTGC  H  <- spaces added for clarity
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    i, alignment = 0, ''
    for letter in s1:
        if i < len(s2) and letter == s2[i]:
            alignment += s2[i]
            i += 1
        else:
            alignment += '-'
    
    return alignment


def mask_aa_coords(pdb_file):
    """
    Gets a mask such that each residue in the amino acid sequence with a 
    coordinate is 1 and otherwise 0.
    """
    full_seqs = aa_seq(pdb_file)
    subseqs = aa_seq_from_coords(pdb_file)
    
    masks = {}
    for chain_id in full_seqs:
        if chain_id in subseqs:
            alignment = align_strings(full_seqs[chain_id], subseqs[chain_id])
            masks[chain_id] = [a != '-' for a in alignment]
        else:
            msg = ('WARNING: Chain ID mismatch between the full sequences and the '
                   'sequences derived from coordinates in the {} file. Chain {} in '
                   'the full sequence is not in the sequences derived from ' 
                   'coordinates. Skipping chain {}')
            warnings.warn(msg.format(pdb_file, chain_id, chain_id))
    return masks


if __name__ == '__main__':
    def main():
        pdb_file = './pdbs/6mxs.pdb' 
        for file in listdir('pdbs/'):
            if file[-4:] != '.pdb':
                continue

            pdb_file = 'pdbs/{}'.format(file)
            print(pdb_file)
            seqs = aa_seq(pdb_file)
            incomplete_seqs = aa_seq_from_coords(pdb_file)
            print(incomplete_seqs)

            k = list(seqs.keys())[0]
            print(k)
            print(seqs['H'])
            print(incomplete_seqs[k])
            print(align_strings(seqs['H'], incomplete_seqs[k]), end='\n\n')
            #print(CB_coords(pdb_file)['H'])
    main()

