
# %%
import os
import argparse
import pandas as pd
import torch
from LamNet import LamNet
from dataset import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from itertools import product

# Set up device (GPU/MPS/CPU)
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print("cuda ok")
elif torch.backends.mps.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = "mps"
    print("mps ok")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "cpu"

# %%

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--cutoff', default=10, type=int)
    parser.add_argument('--task_type', default='rbfe', type=str, help='rbfe, abfe')
    parser.add_argument('--mode', default='optimize', type=str, help='optimize')
    parser.add_argument('--system', default='CDK2_1oiu_1h1q', type=str)
    parser.add_argument('--model_path', default=None, type=str)

    args = parser.parse_args()

    # Set up paths
    data_dir = f'./data/{args.task_type}'
    wk_dir = f'./{args.mode}/{args.task_type}'
    
    protein = (args.system).split('_')[0]
    sys_dir = os.path.join(wk_dir, protein)
    
    # Load data
    df = pd.read_csv(os.path.join(wk_dir, f'{args.system}.csv'))

    # Initialize dataset and dataloader
    dataset = GraphDataset(sys_dir, df, create=True, task_type=args.task_type, mode=args.mode)
    dataloader = PLIDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Set up device
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

    # Extract data from dataframe
    pro_list = []
    lig1_list = []
    lig2_list = []
    lambdai_list = []
    for i, row in df.iterrows():
        pro, lig1, lig2, lambdai =  \
        row['protein'], row['ligand1'], row['ligand2'], float(row['lambdai'])
        pro_list.append([pro])
        lig1_list.append([lig1])
        lig2_list.append([lig2])
        lambdai_list.append([lambdai])

    # %%
    # Make predictions
    pred_list = []
    for batch in dataloader:
        graph1, graph2, extra, _ = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        extra = extra.to(device)
        with torch.no_grad():
            pred = model(graph1, graph2, extra)
            pred_list.append(pred.detach().cpu().numpy())             
    pred = np.concatenate(pred_list, axis=0)

    # Save raw predictions
    res1 = pd.DataFrame()
    res1["protein"] = np.array(pro_list).reshape(-1)
    res1["ligand1"] = np.array(lig1_list).reshape(-1)
    res1["ligand2"] = np.array(lig2_list).reshape(-1)
    res1["lambdai"] = np.array(lambdai_list).reshape(-1)
    res1["y_pred"] = np.array(pred).reshape(-1)
    res1.to_csv(f"{sys_dir}/{args.system}_result.csv")

    # %%
    # Load predictions and optimize lambda parameters
    Gi_df = pd.read_csv(os.path.join(sys_dir, f'{args.system}_result.csv'))

    pro_list2 = []
    lig1_list2 = []
    lig2_list2 = []
    param_list = []
    Gi_list = []
    count = 0

    # Process first half of data (forward direction)
    for i, row in Gi_df.iterrows():
        pro, lig1, lig2, lambdai, Gi =  \
        row['protein'], row['ligand1'], row['ligand2'], float(row['lambdai']), float(row['y_pred'])
        
        if 0 <= i <= 500:  # Forward direction
            if lambdai == 0 or lambdai == 0.5:  # Always keep endpoints
                pro_list2.append([pro])
                lig1_list2.append([lig1])
                lig2_list2.append([lig2])
                param_list.append([lambdai])
                Gi_list.append([Gi])
                count += 1
            else:
                # Check overlap with previous point
                overlap = abs(Gi - float(Gi_list[count-1][0]))
                if overlap <= args.cutoff:  # Keep point if overlap is small enough
                    if len(param_list) < count+1:
                        param_list.append([lambdai])
                        Gi_list.append([Gi])
                        pro_list2.append([pro])
                        lig1_list2.append([lig1])
                        lig2_list2.append([lig2])
                    if len(param_list) == count+1:
                        param_list[count] = [lambdai]
                        Gi_list[count] = [Gi]
                        pro_list2[count] = [pro]
                        lig1_list2[count] = [lig1]
                        lig2_list2[count] = [lig2]
                        # Remove point if it's too close to endpoint
                        if lambdai == 0.001:
                            param_list.pop()
                            Gi_list.pop()
                            pro_list2.pop()
                            lig1_list2.pop()
                            lig2_list2.pop()
                if overlap > args.cutoff:
                    count += 1

        # Process second half (backward direction)
        if 500 < i <= 1001:
            if lambdai == 0.5 or lambdai == 0:  # Always keep endpoints
                count = len(param_list)
                pro_list2.append([pro])
                lig1_list2.append([lig1])
                lig2_list2.append([lig2])
                param_list.append([1-lambdai])  # Reverse lambda for backward direction
                Gi_list.append([Gi])
                count += 1
            else:
                overlap = abs(Gi - float(Gi_list[count-1][0]))
                if overlap <= args.cutoff:
                    if len(param_list) < count+1:
                        param_list.append([1-lambdai])
                        Gi_list.append([Gi])
                        pro_list2.append([pro])
                        lig1_list2.append([lig1])
                        lig2_list2.append([lig2])
                    if len(param_list) == count+1:
                        param_list[count] = [1-lambdai]
                        Gi_list[count] = [Gi]
                        pro_list2[count] = [pro]
                        lig1_list2[count] = [lig1]
                        lig2_list2[count] = [lig2]
                        if lambdai == 0.001:
                            param_list.pop()
                            Gi_list.pop()
                            pro_list2.pop()
                            lig1_list2.pop()
                            lig2_list2.pop()
                if overlap > args.cutoff:
                    count += 1

    # Save optimized parameters
    res2 = pd.DataFrame()
    res2["protein"] = np.array(pro_list2).reshape(-1)
    res2["ligand1"] = np.array(lig1_list2).reshape(-1)
    res2["ligand2"] = np.array(lig2_list2).reshape(-1)
    res2["lambdai"] = np.array(param_list).reshape(-1)
    res2["Gi"] = np.array(Gi_list).reshape(-1)
    res2.to_csv(f"{sys_dir}/{args.system}_param.csv")

