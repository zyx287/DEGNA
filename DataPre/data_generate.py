import argparse
from torch.utils.data import DataLoader
import shutil
import os

from raw_features import RawFeatureExtraction
'''
Generate HHblits output file
'''
def extract_raw_features(raw_feat: RawFeatureExtraction):
    """
    Featurize proteins, ligands and complexes
    :param raw_feat: The input RawFeatureExtraction class
    """
    print("Extracting protein sequences from the PDB file...\n")
    raw_feat.extract_sequences()
    print("Getting coordinates of heavy atoms in the protein...\n")
    raw_feat.get_protein_ha_coordinates()
    print("Extracting ligand raw features...\n")
    raw_feat.get_ligand_info()
    print("Computing distance matrices of the complex...\n")
    raw_feat.get_distance_matrix()
    print("Running HHblits and extracting protein raw features...\n")
    raw_feat.get_residue_features()

def batch_process_files(root_dir):
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if os.path.isdir(file_path):
            protein_path = os.path.join(file_path,f"{filename}_protein.pdb")
            ligand_path = os.path.join(file_path,f"{filename}_ligand.mol2")
            TMP_DATA_DIR=file_path
            args=["$HHblits Database dir",f"{protein_path}",f"{ligand_path}","mol2","4",f"{TMP_DATA_DIR}"]
            rf = RawFeatureExtraction(args[0], args[1], args[2], args[3], args[4], args[5])
            extract_raw_features(rf)
            print(f"Processing {filename}")
        else:
            process_file(file_path)

def process_file(file_path):
    print(f"This isn't a file: {file_path}")

if __name__ == "__main__":
    # Using batch_process_files() to process all file in the root dir
    root_dir = 'root dir with dataset'
    batch_process_files(root_dir)
