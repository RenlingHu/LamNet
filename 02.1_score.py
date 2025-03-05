
# %%
import os
import torch
import argparse

# Set up CUDA device if available
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print("cuda ok")
elif torch.backends.mps.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "mps"
    print("mps ok")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "cpu"

import pandas as pd
from LamNet import LamNet
from dataset import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from itertools import product

# %%

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--task_type', default='rbfe', type=str, help='rbfe, abfe')
    parser.add_argument('--mode', default='score', type=str, help='score')
    parser.add_argument('--system', default='PTP1B', type=str)
    parser.add_argument('--connect', default='weak', type=str, help='weak, strong, custom')
    parser.add_argument('--model_path', default=None, type=str)

    args = parser.parse_args()

    # Set up paths
    data_dir = f'./data/{args.task_type}'
    wk_dir = f'./{args.mode}/{args.task_type}'
    sys_dir = os.path.join(wk_dir, args.system)
    
    # Load data based on task type and connection type
    if args.task_type == 'rbfe':
        if args.connect == 'weak':
            df = pd.read_csv(os.path.join(wk_dir, f'{args.system}-weak.csv')) # Weak-connected graph
        elif args.connect == 'strong':
            df = pd.read_csv(os.path.join(wk_dir, f'{args.system}-strong.csv')) # Strong-connected graph
        else:
            df = pd.read_csv(os.path.join(wk_dir, f'{args.system}(ladybugs).csv')) # LadyBUGS
    if args.task_type == 'abfe':
        df = pd.read_csv(os.path.join(wk_dir, f'{args.system}.csv'))

    # Initialize dataset and dataloader
    dataset = GraphDataset(sys_dir, df, create=True, task_type=args.task_type, mode=args.mode)
    dataloader = PLIDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set up device (GPU/MPS/CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    # Initialize and load model
    model = LamNet(35, 1, 256).to(device)
    load_model_dict(model, args.model_path)
    model.eval()

    # Initialize lists to store predictions and metadata
    pred_list = []
    pro_list = []
    lig1_list = []
    lig2_list = []
    lambdai_list = []

    # Extract data from dataframe
    for i, row in df.iterrows():
        if args.task_type == 'rbfe':
            pro, lig1, lig2, lambdai =  \
            row['protein'], row['ligand1'], row['ligand2'], float(row['lambdai'])
            lig2_list.append([lig2])
        if args.task_type == 'abfe':
            pro, lig1, lambdai =  \
            row['protein'], row['ligand'], float(row['lambdai'])
        
        pro_list.append([pro])
        lig1_list.append([lig1])
        lambdai_list.append([lambdai])

    # %%
    # Make predictions
    for batch in dataloader:
        graph1, graph2, extra, _ = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        extra = extra.to(device)
        with torch.no_grad():
            pred = model(graph1, graph2, extra)
            pred_list.append(pred.detach().cpu().numpy())
    pred = np.concatenate(pred_list, axis=0)

    # Create first results dataframe with raw predictions
    res1 = pd.DataFrame()
    res1["protein"] = np.array(pro_list).reshape(-1)
    res1["ligand1"] = np.array(lig1_list).reshape(-1)
    if args.task_type == 'rbfe':  
        res1["ligand2"] = np.array(lig2_list).reshape(-1)
    res1["lambdai"] = np.array(lambdai_list).reshape(-1)
    res1["y1_pred"] = np.array(pred_list).reshape(-1)

    # %%
    # Initialize lists for scoring
    score_list = []
    pro_list2 = []
    lig1_list2 = []
    lig2_list2 = []
    leg1_list = []
    leg2_list = []

    # Calculate scores from predictions
    for i in range(len(pred_list)):
        if i%2 == 0 :
            pro_list2.append(pro_list[i])
            lig1_list2.append(lig1_list[i])
            if args.task_type == 'rbfe':
                lig2_list2.append(lig2_list[i])
            leg1_list.append(pred_list[i])
            leg2_list.append(pred_list[i+1])
        if i%2 == 1:
            if args.task_type == 'rbfe':
                score_list.append([pred_list[i]-pred_list[i-1]])
            if args.task_type == 'abfe':
                # Add correction terms for absolute binding free energy
                score_list.append([pred_list[i]-pred_list[i-1]+0.87-0.2902542])

    # Create second results dataframe with scores
    res2 = pd.DataFrame()
    res2["protein"] = np.array(pro_list2).reshape(-1)
    res2["ligand1"] = np.array(lig1_list2).reshape(-1)
    if args.task_type == 'rbfe':
        res2["ligand2"] = np.array(lig2_list2).reshape(-1)
    res2["LamNet_score"] = np.array(score_list).reshape(-1)
    res2["leg1"] = np.array(leg1_list).reshape(-1)
    res2["leg2"] = np.array(leg2_list).reshape(-1)

    # Save results
    res2.to_csv(f"{wk_dir}/{args.system}_score.csv")