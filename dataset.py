# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    """
    One-hot encoding for a value k among possible values
    Args:
        k: The value to encode
        possible_values: List of possible values
    Returns:
        List of booleans representing one-hot encoding
    """
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    """
    One-hot encoding with unknown value handling
    Args:
        x: The value to encode
        allowable_set: List of allowed values, last value is treated as unknown
    Returns:
        List of booleans representing one-hot encoding
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    """
    Extract atom features and add them to the graph
    Args:
        mol: RDKit molecule object
        graph: NetworkX graph object
        atom_symbols: List of allowed atom symbols
        explicit_H: Whether to include explicit hydrogens
    """
    for atom in mol.GetAtoms():
        # Get atom features including:
        # - Atom type one-hot encoding
        # - Degree one-hot encoding  
        # - Implicit valence one-hot encoding
        # - Hybridization one-hot encoding
        # - Aromaticity flag
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    """
    Add edges to the graph based on molecular bonds
    Args:
        mol: RDKit molecule object
        graph: NetworkX graph object
    """
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j)

def mol2graph(mol):
    """
    Convert molecule to graph representation
    Args:
        mol: RDKit molecule object
    Returns:
        x: Node feature matrix
        edge_index: Edge index matrix
    """
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    """
    Create interaction graph between ligand and pocket
    Args:
        ligand: RDKit molecule object for ligand
        pocket: RDKit molecule object for pocket
        dis_threshold: Distance threshold for interactions
    Returns:
        edge_index_inter: Edge index matrix for interactions
    """
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < float(dis_threshold))
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter



# %%
def graph(path, label1=None, label2=None, save_path=None, dis_threshold=5, graph_type='complex'):
    """
    Process a graph from a given path and save it if save_path is provided.
    parameters:
        path: str, the directory of the input file
        label1: float, the first label of the graph (Gi)
        label2: float, the second label of the graph (aue)
        save_path: str, the path to save the processed graph
        dis_threshold: float, the distance threshold for pocket definition
        graph_type: str, the type of the graph, 'ligand' or 'complex'
    """
    with open(path, 'rb') as f:
        
        if graph_type == 'ligand':
            ligand = pickle.load(f)
            atom_num_l = ligand.GetNumAtoms()
            pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
            x_l, edge_index_l = mol2graph(ligand)
    
            x = x_l
            edge_index_intra = edge_index_l
            pos = pos_l
            split = torch.zeros((atom_num_l, ))
            
            edge_index_inter = torch.empty((2, 0), dtype=torch.long) #None is not allowed
            
            if label1 is not None:
                y1 = torch.FloatTensor([label1])
            if label2 is not None:
                y2 = torch.FloatTensor([label2])
                data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, 
                            y1=y1, y2=y2, pos=pos, split=split)
            else:
                data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, 
                            pos=pos, split=split)
                
        else:  # complex
            ligand, pocket = pickle.load(f)
            atom_num_l = ligand.GetNumAtoms() 
            atom_num_p = pocket.GetNumAtoms()

            pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
            pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())

            x_l, edge_index_l = mol2graph(ligand)
            x_p, edge_index_p = mol2graph(pocket)

            x = torch.cat([x_l, x_p], dim=0)
            edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
            edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)

            pos = torch.concat([pos_l, pos_p], dim=0)
            split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p, ))], dim=0)

            if label1 is not None:
                y1 = torch.FloatTensor([label1])
            if label2 is not None:
                y2 = torch.FloatTensor([label2])
                data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, 
                          y1=y1, y2=y2, pos=pos, split=split)
            else:
                data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
                          pos=pos, split=split)

    if save_path is not None:
        torch.save(data, save_path)
    
    # return x, atom_num_l, pos_l, edge_index_intra
    # return x, atom_num_l, atom_num_p, pos_l, pos_p, edge_index_intra, edge_index_inter


# %%

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    parameters:
        wk_dir: str, the working directory
        data_df: pd.DataFrame, the data frame containing the data
        dis_threshold: float, the distance threshold for pocket definition
        graph_type: str, the type of the graph, 'ligand' or 'complex'
        num_process: int, the number of processes to generate graphs
        create: bool, whether to create the graph
        task_type: str, the type of the task, 'rbfe' or 'abfe'
        mode: str, the mode of the task, 'single', 'multi', 'fewshot', 'score', 'optimize'
    """
    def __init__(self, wk_dir, data_df, dis_threshold=5, num_process=8, create=False, 
                 task_type='rbfe', mode='single'):
        self.wk_dir = wk_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.create = create
        self.graph1_paths = None
        self.graph2_paths = None
        self.extra = None
        self.num_process = num_process
        self.task_type = task_type # 'rbfe' or 'abfe'
        self.mode = mode # 'single', 'multi', 'fewshot', 'score', 'optimize'
        self._pre_process()

    def _pre_process(self):
        wk_dir = self.wk_dir
        data_df = self.data_df
        # dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex1_path_list = []
        complex2_path_list = []
        ligand_path_list = []
        complex1_id_list = []
        complex2_id_list = []
        graph1_path_list = []
        graph2_path_list = []
        pair_id_list = []
        
        protein = []
        extra = []
        label1_list = []
        label2_list = []
                
        for i, row in data_df.iterrows():
            pro = row['protein']
            protein.append([pro])
            if self.task_type == 'rbfe':
                lig1 = row['ligand1']
                lig2 = row['ligand2']
            elif self.task_type == 'abfe':
                lig1 = row['ligand']
                lig2 = None
            lambdai = float(row['lambdai'])
            extra.append([lambdai])

            # Handle labels based on mode
            if self.mode in ['single', 'multi', 'fewshot']:
                label1 = float(row['Gi(kcal/mol)'])
                label1_list.append(label1)
                label2 = float(row['aue(kcal/mol)'])
                label2_list.append(label2)
                
            # Generate pair name based on mode
            if self.task_type == 'rbfe':
                if self.mode in ['single', 'score']:
                    pair = f"{lig1}-{lig2}-{lambdai}"
                elif self.mode in ['multi', 'fewshot']:
                    pair = f"{pro}-{lig1}-{lig2}-{lambdai}"
                elif self.mode == 'optimize':
                    pair = f"{lig1}-{lig2}"
            elif self.task_type == 'abfe':
                if self.mode in ['single', 'score']:
                    pair = f"{lig1}-{lambdai}"
                elif self.mode in ['multi', 'fewshot']:
                    pair = f"{pro}-{lig1}-{lambdai}"
                elif self.mode == 'optimize':
                    pair = f"{lig1}"
                
            pair_id_list.append(pair)
            pair_dir = os.path.join(wk_dir, pair)
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)

            if self.task_type == 'rbfe':
                if self.mode in ['single', 'multi', 'fewshot']:
                    data_file = 'train'
                if self.mode in ['score', 'optimize']:
                    data_file = self.mode
                input_path = f'./data/rbfe/{data_file}'
                command1 = f'cp -r {input_path}/{pro}/{lig1}/{lig1}_pocket.rdkit {pair_dir}/'
                os.system(command=command1)
                command2 = f'cp -r {input_path}/{pro}/{lig2}/{lig2}_pocket.rdkit {pair_dir}/'
                os.system(command=command2)
            elif self.task_type == 'abfe':
                if self.mode in ['single', 'multi', 'fewshot']:
                    data_file = 'train'
                if self.mode in ['score', 'optimize']:
                    data_file = self.mode
                input_path = f'./data/abfe/{data_file}'
                command1 = f'cp -r {input_path}/{pro}/{lig1}/{lig1}.rdkit {pair_dir}/'
                os.system(command=command1)
                command2 = f'cp -r {input_path}/{pro}/{lig1}/{lig1}_pocket.rdkit {pair_dir}/'
                os.system(command=command2)

            # Setup paths
            if self.task_type == 'rbfe':
                graph1_path = os.path.join(pair_dir, f"{lig1}_pocket.pyg")
                graph2_path = os.path.join(pair_dir, f"{lig2}_pocket.pyg")
            elif self.task_type == 'abfe':
                graph1_path = os.path.join(pair_dir, f"{lig1}.pyg")
                graph2_path = os.path.join(pair_dir, f"{lig1}_pocket.pyg")


            if self.task_type == 'rbfe':
                complex1_path = os.path.join(pair_dir, f"{lig1}_pocket.rdkit")
                complex1_path_list.append(complex1_path)
                complex1_id_list.append(lig1)
                complex2_path = os.path.join(pair_dir, f"{lig2}_pocket.rdkit")
                complex2_path_list.append(complex2_path)
                complex2_id_list.append(lig2)
            elif self.task_type == 'abfe':
                ligand_path = os.path.join(pair_dir, f"{lig1}.rdkit")
                ligand_path_list.append(ligand_path)
                complex1_path = os.path.join(pair_dir, f"{lig1}_pocket.rdkit")
                complex1_path_list.append(complex1_path)
                complex1_id_list.append(lig1)

            graph1_path_list.append(graph1_path)
            graph2_path_list.append(graph2_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            
            # Generate graphs based on task type and mode
            if self.task_type == 'abfe':
                if self.mode in ['single', 'multi', 'fewshot']:
                    pool.starmap(graph,
                               [(p, l1, l2, g, self.dis_threshold, 'ligand') for p, l1, l2, g in zip(ligand_path_list, label1_list, label2_list, graph1_path_list)])
                    pool.starmap(graph,
                               [(p, l1, l2, g, self.dis_threshold, 'complex') for p, l1, l2, g in zip(complex1_path_list, label1_list, label2_list, graph2_path_list)])
                elif self.mode in ['score', 'optimize']:
                    pool.starmap(graph,
                               [(p, None, None, g, self.dis_threshold, 'ligand') for p, g in zip(ligand_path_list, graph1_path_list)])
                    pool.starmap(graph,
                               [(p, None, None, g, self.dis_threshold, 'complex') for p, g in zip(complex1_path_list, graph2_path_list)])
            else:  # rbfe mode
                if self.mode in ['score', 'optimize']:
                    pool.starmap(graph,
                               [(p, None, None, g, self.dis_threshold, 'complex') for p, g in zip(complex1_path_list, graph1_path_list)])
                    pool.starmap(graph,
                               [(p, None, None, g, self.dis_threshold, 'complex') for p, g in zip(complex2_path_list, graph2_path_list)])
                elif self.mode in ['single', 'multi', 'fewshot']:
                    pool.starmap(graph,
                               [(p, l1, l2, g, self.dis_threshold, 'complex') for p, l1, l2, g in zip(complex1_path_list, label1_list, label2_list, graph1_path_list)])
                    pool.starmap(graph,
                               [(p, l1, l2, g, self.dis_threshold, 'complex') for p, l1, l2, g in zip(complex2_path_list, label1_list, label2_list, graph2_path_list)])
            pool.close()
            pool.join()

        self.graph1_paths = graph1_path_list
        self.graph2_paths = graph2_path_list
        self.extra = extra
        self.protein = protein

    def __getitem__(self, idx):
        return torch.load(self.graph1_paths[idx]), torch.load(self.graph2_paths[idx]), self.extra[idx], self.protein[idx]

    def __len__(self):
        return len(self.data_df)


def collate_fn(data):
    graph1, graph2, extra, protein = zip(*data)
    data1 = Batch.from_data_list(graph1)
    data2 = Batch.from_data_list(graph2)
    data3 = torch.tensor(extra)
    
    protein_values_list = [list(map(ord, pro[0])) for pro in protein]
    max_length = 10
    truncated_protein_values_list = [protein[:max_length] for protein in protein_values_list]
    padded_protein_values_list = [protein + [0] * (max_length - len(protein)) for protein in truncated_protein_values_list]
    data4 = torch.tensor(padded_protein_values_list, dtype=torch.long)
    return data1, data2, data3, data4

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=collate_fn, **kwargs)

