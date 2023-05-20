import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from data import *

class MultiLoader(Dataset):
    """ 
    Combine multiple Binding Data Instances
    data_list: Process each PDBID folder and append the features and label to the data_list
    """
    def __init__(self, data_list: list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> Tuple:
        return self.data_list[index]
    
def process_folder(folder_path: str, label_df: pd.DataFrame) -> Tuple[List, List]:
    """
    Process the data in a single folder and return the features and labels
    :param folder_path: The path of the folder contains the data
    :param label_df: The dataframe contains the labels
    :return: A tuple contains the features and labels
    """
    # Load features
    prot_feat_path = os.path.join(folder_path, "protein_features.pkl")
    lig_info_path = os.path.join(folder_path, "ligand_info.pkl")
    dist_info_path = os.path.join(folder_path, "distance_info.pkl")
    dataset = BindingData(prot_feat_path, lig_info_path, dist_info_path)

    # Get the corresponding label (pKd) from df (read the tables)
    pdb_id = os.path.basename(folder_path)
    label = label_df.loc[label_df['pdb_id'] == pdb_id, 'pkd']

    #Reture the features and label as a tuple
    return dataset.input_list, label

def dataset_generator(data_folder_path: str, label_df_path: str):
    '''
    :parm data_folder_path: The path of ROOT dir of different PDBID named folder
    :parm label_df_path: The path of index table contains pKd
    :return: A Processed data for dataloder (MultiLoader)
    '''
    data_folder = data_folder_path
    pdb_folders = glob.glob(os.path.join(data_folder, "*"))
    label_df = pd.read_csv(label_df_path)

    data_list = []

    for pdb_folder in pdb_folders:
        processed_data = process_folder(pdb_folder, label_df)
        data_list.append(processed_data)
    
    dataset = MultiLoader(data_list)
    
    return dataset