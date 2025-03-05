# %%
import os
import argparse
import torch.utils

# Set up CUDA device if available
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print("cuda ok")
elif torch.backends.mps.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "mps"
    print("mps ok")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "cpu"

import torch
import torch.nn as nn
import torch.optim as optim
import random
torch.autograd.set_detect_anomaly(True)

import pandas as pd
from utils import AverageMeter
from LamNet import LamNet
from dataset import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from logger import TrainLogger

def setup_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', default=True, type=bool)
    parser.add_argument('--task_type', default='rbfe', type=str, help='rbfe')
    parser.add_argument('--mode', default='single', type=str, help='single')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--test_system', \
        default=['CDK2', 'JNK1', 'TYK2', 'hif2a', \
                 'p38', 'pfkfb3', 'syk', 'tnks2', \
                 'BACE', 'MCL1', 'PTP1B', 'thrombin'], \
        type=list)
    args = parser.parse_args()

    # Set up device (GPU/MPS/CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # Initialize model
    model = LamNet(35, 1, 256).to(device)

    def generalize(data_dir, wk_dir, system, model, device):
        """
        Evaluate model generalization performance on a specific system
        
        Args:
            data_dir: Data directory path
            wk_dir: Working directory path(single)  
            system: Target system name
            model: model
            device: Computing device
            
        Returns:
            rmse: Root mean squared error
            coff: Correlation coefficient
        """
        sys_dir = os.path.join(wk_dir, system)
        df = pd.read_csv(os.path.join(data_dir, f'{system}-single-valid.csv'))
        dataset = GraphDataset(sys_dir, df, create=True, task_type=args.task_type, mode=args.mode)
        dataloader = PLIDataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        
        pred_list = []
        label1_list = []
        label2_list = []

        # Make predictions on validation set
        for batch in dataloader:
            graph1, graph2, extra, _ = batch
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            extra = extra.to(device)
            with torch.no_grad():
                pred = model(graph1, graph2, extra)
                label1 = graph1.y1
                label2 = graph2.y2
                
                pred_list.append(pred.detach().cpu().numpy().flatten())
                label1_list.append(label1.detach().cpu().numpy())
                
        # Calculate metrics
        pred = np.concatenate(pred_list, axis=0)
        label1 = np.concatenate(label1_list, axis=0)

        coff = np.corrcoef(pred, label1)[0, 1]
        rmse = np.sqrt(mean_squared_error(label1, pred))

        return rmse, coff
    
    # Set up paths
    data_dir = f'./data/{args.task_type}'
    wk_dir = f'./{args.mode}/{args.task_type}'

    # Load model and set to eval mode
    load_model_dict(model, args.model_path)
    model = model.to(device)
    model.eval()
    
    # Set random seed
    setup_seed(1229)

    # %%
    # Evaluate on each test system
    for system in args.test_system:
        rmse, coff = generalize(data_dir=data_dir, wk_dir=wk_dir, system=system, model=model, device=device)
        print(f"{system}_valid_rmse-%.4f, {system}_valid_r-%.4f" % (rmse, coff))