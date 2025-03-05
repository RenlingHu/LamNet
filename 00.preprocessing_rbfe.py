# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
import subprocess
import argparse
RDLogger.DisableLog('rdApp.*')
from Bio import PDB


def std_protein(input_pdb, output_pdb):
    """
    Convert non-standard amino acid residues to standard ones
    
    Args:
        input_pdb: Input PDB file path
        output_pdb: Output PDB file path
    """
    NON_STANDARD_TO_STANDARD = {
        "HIE": "HIS",  # Deprotonated at delta position
        "HID": "HIS",  # Deprotonated at epsilon position  
        "HIP": "HIS",  # Double protonated
        "LYN": "LYS",  # Deprotonated
        "ASH": "ASP",  # Protonated
        "GLH": "GLU",  # Protonated
        "CYX": "CYS",  # Disulfide bonded
        "CYM": "CYS",  # Deprotonated
        "TYM": "TYR",  # Deprotonated
        "GLZ": "GLY",  # Synonymous coding
        "ALY": "LYS",  # Modified
        "MSE": "MET",  # Selenomethionine
        "PTR": "TYR",  # Phosphorylated
        "SEP": "SER",  # Phosphorylated
        "TPO": "THR",  # Phosphorylated
        "CSO": "CYS",  # Oxidized
        "CSS": "CYS",  # Disulfide bond
        "CME": "CYS",  # Carboxymethylated
        "ACE": "ALA",  # C-terminal
        "NMA": "ALA",  # N-terminal
    }

    # Convert non-standard residues to standard residues
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                if resname in NON_STANDARD_TO_STANDARD:
                    print(f"{resname} converted to {NON_STANDARD_TO_STANDARD[resname]}")
                    residue.resname = NON_STANDARD_TO_STANDARD[resname]
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


# %%
def generate_pocket(data_root, distance=5, input_ligand_format='sdf'):
    """
    Generate protein pocket structure around ligand binding site
    
    Args:
        data_root: Directory containing unprocessed data
        distance: Distance cutoff (in Angstrom) to define pocket region
        input_ligand_format: Format of input ligand file
    """
    system_id = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]
        
    for sid in system_id:
        system_dir = os.path.join(data_root, sid)
        complex_id = [d for d in os.listdir(system_dir) if os.path.isdir(os.path.join(system_dir, d)) and not d.startswith('.')]
        for cid in complex_id:
            complex_dir = os.path.join(system_dir, cid)
            if os.path.exists(os.path.join(complex_dir, f'{cid}_pocket.pdb')):
                continue
                # pass
            else:
                if input_ligand_format == 'pdb':
                    lig_native_path = os.path.join(complex_dir, f"{cid}.pdb")
                elif input_ligand_format == 'mol2':
                    lig_native_path = os.path.join(complex_dir, f"{cid}.pdb")
                    subprocess.run(['obabel', '-imol2', os.path.join(complex_dir, f"{cid}.mol2"), '-opdb', '-O', lig_native_path])
                elif input_ligand_format == 'sdf':
                    lig_native_path = os.path.join(complex_dir, f"{cid}.pdb")
                    subprocess.run(['obabel', '-isdf', os.path.join(complex_dir, f"{cid}.sdf"), '-opdb', '-O', lig_native_path])
                receptor_name = os.path.basename(system_dir)
                if os.path.exists(os.path.join(complex_dir, f"{receptor_name}.pdb")):
                    protein_path = os.path.join(complex_dir, f"{receptor_name}.pdb")
                    protein_std_path = os.path.join(complex_dir, f"{receptor_name}_std.pdb")
                    std_protein(protein_path, protein_std_path)
                else:
                    protein_path = os.path.join(complex_dir, f"protein.pdb")
                    protein_std_path = os.path.join(complex_dir, f"protein_std.pdb")
                    std_protein(protein_path, protein_std_path)

                pymol.cmd.load(protein_path, "protein")
                pymol.cmd.remove('resn HOH')
                pymol.cmd.load(lig_native_path, "ligand")
                pymol.cmd.remove('hydrogens')
                pymol.cmd.select('Pocket', f'(protein within {distance} of ligand)')
                pymol.cmd.save(os.path.join(complex_dir, f'{cid}_pocket.pdb'), 'Pocket')
                pymol.cmd.delete('all')
                print(f'{cid}-pocket prepared!')


def all2convert(data_dir, lig, input_ligand_format='sdf', output_ligand_format='sdf'):
    """
    Convert ligand between different file formats using OpenBabel
    
    Args:
        data_dir: Directory containing unprocessed data
        lig: Ligand identifier
        input_ligand_format: Input file format
        output_ligand_format: Output file format
    
    Returns:
        ligand_dir: Directory containing converted ligand
        ligand_path: Path to converted ligand file
    """
    ligand_dir= os.path.join(data_dir, lig)
    if output_ligand_format == 'pdb':
        ligand_path = os.path.join(ligand_dir, f'{lig}.pdb')
        if input_ligand_format == 'sdf':
            subprocess.run(['obabel', '-isdf', os.path.join(ligand_dir, f"{lig}.sdf"), '-opdb', '-O', ligand_path])
        elif input_ligand_format == 'mol2':
            subprocess.run(['obabel', '-imol2', os.path.join(ligand_dir, f"{lig}.mol2"), '-opdb', '-O', ligand_path])
    elif output_ligand_format == 'mol2':
        ligand_path = os.path.join(ligand_dir, f'{lig}.mol2')
        if input_ligand_format == 'pdb':
            subprocess.run(['obabel', '-ipdb', os.path.join(ligand_dir, f"{lig}.pdb"), '-omol2', '-O', ligand_path])
        elif input_ligand_format == 'sdf':
            subprocess.run(['obabel', '-isdf', os.path.join(ligand_dir, f"{lig}.sdf"), '-omol2', '-O', ligand_path])
    elif output_ligand_format == 'sdf':
        ligand_path = os.path.join(ligand_dir, f'{lig}.sdf')
        if input_ligand_format == 'pdb':
            subprocess.run(['obabel', '-ipdb', os.path.join(ligand_dir, f"{lig}.pdb"), '-osdf', '-O', ligand_path])
        elif input_ligand_format == 'mol2':
            subprocess.run(['obabel', '-imol2', os.path.join(ligand_dir, f"{lig}.mol2"), '-osdf', '-O', ligand_path])
    return ligand_dir, ligand_path


def complex(ligand_dir, ligand_path, pocket_path, cid, input_ligand_format='sdf'):
    """
    Create and save ligand-pocket complex in RDKit format with error handling
    
    Args:
        ligand_dir: Directory to save complex
        ligand_path: Path to ligand file
        pocket_path: Path to pocket file
        cid: Complex identifier
        input_ligand_format: Format of input ligand file
    """
    save_path = os.path.join(ligand_dir, f"{cid}_pocket.rdkit")
    error_path = "error_records.log"

    # Load and validate ligand
    try:
        if input_ligand_format == 'sdf':
            ligand = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=False)[0]
        elif input_ligand_format == 'pdb':
            ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True, sanitize=False)
        if ligand is None:
            with open(error_path, 'a') as f:
                f.write(f"Unable to read {input_ligand_format} file for {cid}\n")
            return
    except Exception as e:
        with open(error_path, 'a') as f:
            f.write(f"Error reading {input_ligand_format} file for {cid}: {str(e)}\n")
        return
        
    if ligand == None:
        with open(error_path, 'a') as f:
            f.write(f"Unable to process {input_ligand_format} of {cid}\n")
        return

    # Load and validate pocket
    try:
        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=False)
        if pocket is None:
            if not os.path.exists(pocket_path):
                with open(error_path, 'a') as f:
                    f.write(f"Pocket file does not exist for {cid}: {pocket_path}\n")
                return
            if os.path.getsize(pocket_path) == 0:
                with open(error_path, 'a') as f:
                    f.write(f"Pocket file is empty for {cid}: {pocket_path}\n") 
                return    
            with open(error_path, 'a') as f:
                f.write(f"Unable to process pocket of {cid}, please check PDB format\n")
            return
            
    except OSError as e:
        with open(error_path, 'a') as f:
            f.write(f"Error reading pocket file for {cid}: {str(e)}\n")
        return

    # Check for 3D conformers
    if not ligand.GetNumConformers() or not pocket.GetNumConformers():
        with open(error_path, 'a') as f:
            f.write(f"Missing 3D coordinates for {cid}\n")
        return

    # Check atom counts
    if ligand.GetNumAtoms() == 0 or pocket.GetNumAtoms() == 0:
        with open(error_path, 'a') as f:
            f.write(f"Empty molecule found for {cid}\n")
        return
    
    # Try to sanitize molecules
    try:
        Chem.SanitizeMol(ligand, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
        Chem.SanitizeMol(pocket, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_PROPERTIES)
    except Exception as e:
        try:
            Chem.SanitizeMol(ligand, sanitizeOps=Chem.SANITIZE_FINDRADICALS|Chem.SANITIZE_SETAROMATICITY|Chem.SANITIZE_SETCONJUGATION)
            Chem.SanitizeMol(pocket, sanitizeOps=Chem.SANITIZE_FINDRADICALS|Chem.SANITIZE_SETAROMATICITY|Chem.SANITIZE_SETCONJUGATION)
        except Exception as e:
            with open(error_path, 'a') as f:
                f.write(f"Failed to sanitize molecules for {cid}: {str(e)}\n")
            return

    # Save complex
    complex = (ligand, pocket)
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)
    except Exception as e:
        with open(error_path, 'a') as f:
            f.write(f"Error saving complex for {cid}: {str(e)}\n")


def generate_complex(data_root, data_df, task_type='abfe', input_ligand_format='pdb', output_ligand_format='sdf'):
    """
    Generate ligand-pocket complexes for all entries in dataset
    
    Args:
        data_root: Directory containing unprocessed data
        data_df: DataFrame with complex information
        task_type: Type of task (abfe or rbfe)
        input_ligand_format: Format of input ligand files
        output_ligand_format: Format to convert ligands to
    """
    system_id = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')]
        
    for sid in system_id:
        data_dir = os.path.join(data_root, sid)
        pbar = tqdm(total=len(data_df))
        for i, row in data_df.iterrows():
            if row['protein'] != sid:
                continue
            lig1 = row['ligand1']
            ligand_dir, ligand_path = all2convert(data_dir, str(lig1), input_ligand_format=input_ligand_format, output_ligand_format=output_ligand_format)
            pocket1_path = os.path.join(data_dir, str(lig1), f'{str(lig1)}_pocket.pdb')
            if not os.path.exists(os.path.join(ligand_dir, f'{str(lig1)}_pocket.rdkit')):
                complex(ligand_dir, ligand_path, pocket1_path, str(lig1), input_ligand_format=output_ligand_format)

            if task_type == 'rbfe':
                lig2 = row['ligand2']
                ligand_dir, ligand_path = all2convert(data_dir, str(lig2), input_ligand_format=input_ligand_format, output_ligand_format=output_ligand_format)
                pocket2_path = os.path.join(data_dir, str(lig2), f'{str(lig2)}_pocket.pdb')
                if not os.path.exists(os.path.join(ligand_dir, f'{str(lig2)}_pocket.rdkit')):
                    complex(ligand_dir, ligand_path, pocket2_path, str(lig2), input_ligand_format=output_ligand_format)
            pbar.update(1)


if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser(description='Preprocess RBFE data')
    
    # Basic parameters
    parser.add_argument('--task_type', type=str, default='rbfe', help='abfe, rbfe')
    parser.add_argument('--mode', type=str, default='train', help='train, score, optimize')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data root directory')
    parser.add_argument('--csv_name', type=str, default=None, help='Dataset CSV filename')
                        
    # Pocket generation parameters
    parser.add_argument('--distance', type=float, default=5.0, help='Distance threshold for defining pocket region (Angstrom)')
    parser.add_argument('--input_ligand_format', type=str, default='pdb', help='Input ligand file format')
    parser.add_argument('--output_ligand_format', type=str, default='pdb', help='Output ligand file format')
    
    args = parser.parse_args()
    
    # Set up paths
    data_root = os.path.join(args.data_dir, args.task_type, args.mode)
    wk_root = os.path.join('score', args.task_type)
    
    # Load data
    if args.mode == 'train':
        data_df = pd.read_csv(os.path.join(args.data_dir, args.task_type, f"{args.csv_name}.csv"))
    else:
        data_df = pd.read_csv(os.path.join(wk_root, f"{args.csv_name}.csv"))

    # Generate structures
    generate_pocket(data_root, distance=args.distance, input_ligand_format=args.input_ligand_format)
    generate_complex(data_root, data_df, task_type=args.task_type,
                    input_ligand_format=args.input_ligand_format, 
                    output_ligand_format=args.output_ligand_format)

# %%