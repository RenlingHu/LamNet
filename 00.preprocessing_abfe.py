import os
import pickle
import argparse
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from Bio.PDB import *
from rdkit import RDLogger
import numpy as np
RDLogger.DisableLog('rdApp.*')


# %%
def generate_pocket(data_dir, data_df, distance=5, input_ligand_format='pdb'):
    """
    Generate protein pocket structure around ligand binding site
    
    Args:
        data_dir: Directory containing unprocessed data
        data_df: DataFrame with protein and ligand information
        distance: Distance cutoff (in Angstrom) to define pocket region
        input_ligand_format: Format of input ligand file (default: pdb)
    """
    for i, row in data_df.iterrows():
        receptor = row['protein']
        system_dir = os.path.join(data_dir, receptor)
        lig = row['ligand']
        parser = PDBParser(QUIET=True)

        complex_dir = os.path.join(system_dir, lig)
        if not os.path.isdir(complex_dir):
            continue
            
        lig_path = os.path.join(complex_dir, f"{lig}.{input_ligand_format}")
        receptor_path = os.path.join(complex_dir, "receptor.pdb")
            
        # Skip if pocket file already exists
        if os.path.exists(os.path.join(complex_dir, f'{lig}_pocket.pdb')):
            continue

        if os.path.isfile(receptor_path) and os.path.isfile(lig_path):
            # Load structures
            structure = parser.get_structure('receptor', receptor_path)
            ligand = parser.get_structure('ligand', lig_path)
                
            # Remove water molecules and hydrogen atoms from receptor
            for model in structure:
                for chain in model:
                    for residue in list(chain):
                        if residue.get_resname() in ['HOH', 'WAT']:
                            chain.detach_child(residue.id)
                        for atom in list(residue):
                            if atom.element == 'H':
                                residue.detach_child(atom.id)
            
            # Remove hydrogen atoms from ligand
            for model in ligand:
                for chain in model:
                    for residue in chain:
                        for atom in list(residue):
                            if atom.element == 'H':
                                residue.detach_child(atom.id)
            
            # Get ligand atom coordinates
            ligand_atoms = []
            for atom in ligand.get_atoms():
                ligand_atoms.append(atom.get_coord())
                
            # Find receptor atoms within distance cutoff of ligand atoms
            ns = NeighborSearch(list(structure.get_atoms()))
            pocket_atoms = set()
            for lig_coord in ligand_atoms:
                close_atoms = ns.search(lig_coord, distance)
                for atom in close_atoms:
                    pocket_atoms.add(atom)
                                
            # Save pocket structure
            io = PDBIO()
            io.set_structure(structure)
            class PocketSelect(Select):
                def accept_atom(self, atom):
                    return atom in pocket_atoms
                    
            io.save(os.path.join(complex_dir, f'{lig}_pocket.pdb'), PocketSelect())
            print(f'{lig}-pocket prepared!')


def generate_ligand(data_dir, data_df):
    """
    Convert ligand PDB files to RDKit molecule format
    
    Args:
        data_dir: Directory containing unprocessed data
        data_df: DataFrame with ligand information
    """
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        receptor = row['protein']
        system_dir = os.path.join(data_dir, receptor)
        lig = row['ligand']
        
        ligand_dir = os.path.join(system_dir, lig)
        ligand_path = os.path.join(system_dir, lig, f'{lig}.pdb')
        # ligand_dir, ligand_path = mol2pdb(system_dir, lig)
        
        save_path = os.path.join(ligand_dir, f"{lig}.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand {lig}")
            continue
            
        with open(save_path, 'wb') as f:
            pickle.dump(ligand, f)
            
        pbar.update(1)


def complex(ligand_dir, ligand_path, pocket_path, lig):
    """
    Create complex of ligand and pocket in RDKit format
    
    Args:
        ligand_dir: Directory to save complex
        ligand_path: Path to ligand PDB file
        pocket_path: Path to pocket PDB file 
        lig: Ligand identifier
    """
    save_path = os.path.join(ligand_dir, f"{lig}_pocket.rdkit")
    
    # Load and check ligand
    ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
    if ligand == None:
        print(f"Unable to process ligand of {lig}")
    if ligand.GetNumBonds() == 0:
        print(f"{lig} Error: Molecule has no bonds.")

    # Load and check pocket
    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
    if pocket == None:
        print(f"Unable to process pocket of {lig}")

    # Save complex
    complex = (ligand, pocket)
    with open(save_path, 'wb') as f:
        pickle.dump(complex, f)


def generate_complex(data_dir, data_df, task_type='abfe'):
    """
    Generate ligand-pocket complexes for all entries in dataset
    
    Args:
        data_dir: Directory containing unprocessed data
        data_df: DataFrame with complex information
        task_type: Type of task (default: abfe)
    """
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        receptor = row['protein']
        system_dir = os.path.join(data_dir, receptor)
        lig = row['ligand']
        
        pocket_path = os.path.join(system_dir, lig, f'{lig}_pocket.pdb')
        ligand_dir = os.path.join(system_dir, lig)
        ligand_path = os.path.join(system_dir, lig, f'{lig}.pdb')
        
        complex(ligand_dir, ligand_path, pocket_path, lig)
        pbar.update(1)


if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser(description='Preprocess ABFE data')
    
    # Basic parameters
    parser.add_argument('--task_type', type=str, default='abfe', help='abfe, rbfe')
    parser.add_argument('--mode', type=str, default='score', help='train, score, optimize')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data root directory')
    parser.add_argument('--csv_name', type=str, default=None, help='Dataset CSV filename')
                        
    # Pocket generation parameters
    parser.add_argument('--distance', type=float, default=5.0, help='Distance threshold for defining pocket region (Angstrom)')
    parser.add_argument('--input_ligand_format', type=str, default='pdb', help='Input ligand file format')
    
    args = parser.parse_args()
    
    # Set up paths
    data_dir = os.path.join(args.data_dir, args.task_type, args.mode)
    
    # Load data
    if args.mode == 'train':
        data_df = pd.read_csv(os.path.join(args.data_dir, args.task_type, f"{args.csv_name}.csv"))
    if args.mode in ['score', 'optimize']:
        data_df = pd.read_csv(os.path.join(args.mode, args.task_type, f"{args.csv_name}.csv"))

    # Generate structures
    generate_pocket(data_dir, data_df, distance=args.distance, 
                   input_ligand_format=args.input_ligand_format)
    generate_complex(data_dir, data_df, task_type=args.task_type)
    generate_ligand(data_dir, data_df)
