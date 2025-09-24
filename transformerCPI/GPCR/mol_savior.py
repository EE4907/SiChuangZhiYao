# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 14:19
@author: LiFan Chen
@Filename: mol_featurizer.py
@Software: PyCharm
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding_unk(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        print("SMILES cannot been parsed!")
        return None, None
    #mol = Chem.AddHs(mol)

    if type(mol) != Chem.rdchem.Mol:
        return None, None
    
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

if __name__ == "__main__":

    import os
    import numpy as np
    from word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec

    model = Word2Vec.load("./GPCR/word2vec_30.model")

    input_folder = './data_processed'
    output_folder = './dataset'

    chunk_size = 40000

    for idx in range (1, 23):
        
        filename = 'split_' + str(idx).zfill(3) + '.txt'
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r") as f:
            smiles_list = f.read().strip().split('\n')

        smiles_list = [s.split('|')[0] for s in smiles_list if '.' not in s]
        smiles_list = [s for s in smiles_list if len(s) > 10]
        N = len(smiles_list)

        compounds, adjacencies = [], []

        start = N - (N % chunk_size) - chunk_size

        for i, smiles in enumerate(smiles_list):
            if i < start:
                continue

            if i % 1000 == 0:
                print(f"{filename}: {i + 1}/{N}")
            if type(smiles) != str:
                continue

            atom_feature, adj = mol_features(smiles)
            if atom_feature is not None and adj is not None:
                compounds.append(atom_feature)
                adjacencies.append(adj)
            else:
                print(f"Invalid SMILES at index {i}: {smiles}")

            if (i + 1) % chunk_size == 0 or (i + 1) == N:
                part_id = 98 if (i + 1) % chunk_size == 0 else 99

                base_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(output_folder, base_name)
                os.makedirs(output_dir, exist_ok=True)

                np.save(os.path.join(output_dir, f'compounds_part{part_id}.npy'), np.array(compounds, dtype=object))
                np.save(os.path.join(output_dir, f'adjacencies_part{part_id}.npy'), np.array(adjacencies, dtype=object))

                print(f"Saved chunk {part_id} to {output_dir}/")

                compounds = []
                adjacencies = []

    print("All files processed.")
