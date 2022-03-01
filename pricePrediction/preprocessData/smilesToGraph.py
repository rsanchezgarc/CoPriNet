import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from torch_geometric.data import Data as GraphData

ALLOWED_ATOMIC_NUMS = {6, 7, 8, 9, 15, 16, 17, 35}
MIN_NUM_ATOMS = 3
MAX_NUM_ATOMS = 60

def get_node_feats(mol: Chem.Mol, atom: Chem.Atom):
    '''

    :param mol: The mol from which the node features will be computed. Currently not used
    :param atom: The atom from which the node features will be computed
    :return:
    '''
    atom_num = atom.GetAtomicNum()
    valence = atom.GetTotalValence()
    charge = atom.GetFormalCharge()
    degree = atom.GetDegree()
    is_aromatic = atom.GetIsAromatic()
    return [float(elem) for elem in [atom_num, valence, charge, degree, is_aromatic]]
    # return [float(elem) for elem in [atom_num, valence, charge, is_aromatic]]

# results = one_of_k_encoding_unk(
#     atom.GetSymbol(),
#     [
#         'B',
#         'C',
#         'N',
#         'O',
#         'F',
#         'Si',
#         'P',
#         'S',
#         'Cl',
#         'As',
#         'Se',
#         'Br',
#         'Te',
#         'I',
#         'At',
#         'other'
#     ]) + one_of_k_encoding(atom.GetDegree(),
#                            [0, 1, 2, 3, 4, 5]) + \
#           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
#           one_of_k_encoding_unk(atom.GetHybridization(), [
#               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
#                                 SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
#           ]) + [atom.GetIsAromatic()]

def get_bond_feats(mol: Chem.Mol, bond: Chem.Bond):
    '''

    :param mol: The mol from which the edge features will be computed. Currently not used
    :param bond: The bond from which the edge features will be computed
    :return:
    '''
    if bond.GetIsAromatic():
        bond_num = 1.5
    else:
        bond_num = bond.GetBondType()
        bond_num = float(bond_num)
    if bond_num > 3:
        bond_num = 4

    # return [bond_num ]
    bondInfo = [bond_num, int(bond.GetIsAromatic()), int(bond.GetIsConjugated()), int(bond.IsInRing()) ]
    # bondInfo += [["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"].index(str(bond.GetStereo()))]

    return bondInfo

def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    nodes = []
    edges_index, edges_attr = [], []

    cur_atom_id = 0
    rdkit_atomId_to_atom_id = {}
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        atom2 = mol.GetAtomWithIdx(atom2_idx)
        atomicNum1 = atom1.GetAtomicNum()
        atomicNum2 = atom2.GetAtomicNum()

        if atomicNum1 == 0 or atomicNum2 == 0:
            continue
        if atomicNum1 not in ALLOWED_ATOMIC_NUMS or atomicNum2 not in ALLOWED_ATOMIC_NUMS:
            return None
        if atom1_idx not in rdkit_atomId_to_atom_id:
            rdkit_atomId_to_atom_id[atom1_idx] = cur_atom_id
            nodes.append( get_node_feats(mol, atom1) )
            cur_atom_id += 1
        if atom2_idx not in rdkit_atomId_to_atom_id:
            rdkit_atomId_to_atom_id[atom2_idx] = cur_atom_id
            nodes.append(  get_node_feats(mol, atom2) )
            cur_atom_id += 1
        idx1, idx2 = rdkit_atomId_to_atom_id[atom1_idx], rdkit_atomId_to_atom_id[atom2_idx]
        # print(atom1_idx, idx1, atom2_idx, idx2)
        edges_index.append([idx1, idx2])
        edges_index.append([idx2, idx1])

        bond_feats = get_bond_feats(mol, bond)
        edges_attr.extend([bond_feats, bond_feats])

    x = np.array(nodes, dtype=np.float32)

    if x.shape[0] < MIN_NUM_ATOMS or x.shape[0] > MAX_NUM_ATOMS:
        return None
    edges_index = np.array( edges_index).T.copy().astype( np.int64 )
    edges_attr = np.array(edges_attr, dtype=np.float32)
    graph_dict = dict(x=x, edge_index=edges_index, edge_attr=edges_attr)
    graph = GraphData(**{key: torch.tensor(val) for key, val in graph_dict.items()} )
    return graph

def fromPerGramToPerMMolPrice(price, smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return price * ExactMolWt(mol) / 1000

def compute_nodes_degree(graphs, max_degree=7):
    from torch_geometric.utils import degree
    import torch
    deg = torch.zeros(max_degree, dtype=torch.long)
    if graphs is None:
        return deg
    for data in graphs:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())[:deg.numel()]
    return deg

def test(smi = "C1=CC=C(C=C1)CC(C(=O)O)N"):

    mol = Chem.MolFromSmiles(smi)
    # print(mol)
    graph = smiles_to_graph(smi)
    print(graph.keys)
    print(graph.x)
    print(graph.edge_index)
    print( graph.edges_attr )
    print( graph.x.shape, graph.edge_index.shape, graph.edges_attr.shape)
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    from matplotlib import pyplot as plt; from rdkit.Chem import Draw; plt.imshow(Draw.MolsToGridImage([mol], molsPerRow=2)); plt.show()

if __name__ == "__main__":
    smis_list = ["[C@H](C)1CCCO1", "O[C@@H](N)C", "C1=CC=C(C=C1)CC(C(=O)O)N"]
    for smi in smis_list:
        test(smi)