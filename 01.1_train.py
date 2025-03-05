
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
from tqdm import tqdm

# %%
def setup_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val(model, dataloader, device):
    """
    Validate model performance
    
    Args:
        model: model
        dataloader: Validation data loader
        device: Computing device
        
    Returns:
        rmse: Root mean squared error
        coff: Correlation coefficient
        pred: Model predictions
        label1: Ground truth labels
        criterion: RMSE at lambda=0.5
    """
    model.eval()

    pred_list = []
    label1_list = []
    label2_list = []
    extra_list = []
    for batch in dataloader:
        graph1, graph2, extra, protein_values = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        extra = extra.to(device)
        protein_values = protein_values.to(device)
        with torch.no_grad():
            pred = model(graph1, graph2, extra)
            label1 = graph1.y1
            label2 = graph1.y2

            pred_list.append(pred.detach().cpu().numpy())
            label1_list.append(label1.detach().cpu().numpy())
            extra_list.append(extra.detach().cpu().numpy())
    pred = np.concatenate(pred_list, axis=0)
    label1 = np.concatenate(label1_list, axis=0)
    extra = np.concatenate(extra_list, axis=0)
        
    coff = np.corrcoef(pred, label1)[0, 1]
    rmse = np.sqrt(mean_squared_error(label1, pred))

    # Calculate RMSE at lambda=0.5
    extra_mask = (extra == 0.5).squeeze()
    pred_lambda05 = pred[extra_mask]
    label1_lambda05 = label1[extra_mask]
    criterion = np.sqrt(mean_squared_error(label1_lambda05, pred_lambda05))

    model.train()

    return rmse, coff, pred, label1, criterion


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--create', default=True, type=bool)
    parser.add_argument('--save_dir', default='./model', type=str)
    parser.add_argument('--max_checkpoints', default=10, type=int)

    parser.add_argument('--batch_size', default=32, type=int) #For multi mode
    # parser.add_argument('--batch_size', default=16, type=int) #For single mode
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--repeat', default=3, type=int)
    parser.add_argument('--early_stop_epoch', default=100, type=int)

    parser.add_argument('--task_type', default='rbfe', type=str, help='rbfe, abfe')
    parser.add_argument('--mode', default='multi', type=str, help='single, multi, fewshot')
    parser.add_argument('--system', default='PTP1B', type=str)
    parser.add_argument('--app_mode', default='score', type=str, help='score, optimize')
    
    parser.add_argument('--lr_single', default=5e-4, type=float, help='learning rate for single mode')
    parser.add_argument('--lr_multi', default=1e-2, type=float, help='learning rate for multi mode')
    parser.add_argument('--weight_decay_single', default=5e-3, type=float, help='weight decay for single mode')
    parser.add_argument('--weight_decay_multi', default=1e-3, type=float, help='weight decay for multi mode')
    
    parser.add_argument('--use_aue_weight', default=True, type=bool, help='whether to use aue as weight')
    
    parser.add_argument('--use_specific_fewshot', default=False, type=bool, help='whether to increase weight of specific target data')
    parser.add_argument('--specific_system', default=None, type=str or list, help='hif2a-nnp or [CDK2-nnp, CDK2]')
    
    args = parser.parse_args()

    setup_seed(1229)

    if args.use_aue_weight:
        print("use aue as weight")
    else:
        print("NOT use aue as weight")
        

    for repeat in range(args.repeat):

        data_dir = f'./data/{args.task_type}'
        wk_dir = f'./{args.mode}/{args.task_type}'
        
        train_dir = os.path.join(wk_dir, args.system)
        valid_dir = os.path.join(wk_dir, args.system)

        # Load training and validation data based on mode
        if args.mode == 'multi':
            train_df = pd.read_csv(os.path.join(data_dir, f'train.csv'))
            valid_df = pd.read_csv(os.path.join(data_dir, f'valid.csv'))
            # train_df = pd.read_csv(os.path.join(data_dir, f'{args.system}-train.csv'))
            # valid_df = pd.read_csv(os.path.join(data_dir, f'{args.system}-valid.csv'))
        if args.mode == 'fewshot':
            train_df = pd.read_csv(os.path.join(data_dir, f'{args.system}-fewshot.csv'))
            valid_df = pd.read_csv(os.path.join(data_dir, f'valid.csv'))
        if args.mode == 'single':
            train_df = pd.read_csv(os.path.join(data_dir, f'{args.system}-single-train.csv'))
            valid_df = pd.read_csv(os.path.join(data_dir, f'{args.system}-single-valid.csv'))

        train_set = GraphDataset(train_dir, train_df, create=True, task_type=args.task_type, mode=args.mode)
        valid_set = GraphDataset(valid_dir, valid_df, create=True, task_type=args.task_type, mode=args.mode)

        train_loader = PLIDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_loader = PLIDataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        logger = TrainLogger(args, create=True, repeat=repeat)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")

        # Set device (CUDA/MPS/CPU)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
        model = LamNet(35, 1, 256).to(device)

        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
        
        # Set optimizer based on mode
        if args.mode == 'single':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_single, weight_decay=args.weight_decay_single)
        elif args.mode in ['multi', 'fewshot']:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_multi, weight_decay=args.weight_decay_multi)
            
        # Learning rate schedulers
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=20,
            min_lr=1e-8,
            verbose=True
        )

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.0001,
            end_factor=1.0,
            total_iters=50
        )

        # Loss function
        if args.use_aue_weight or args.use_specific_fewshot:
            losscalculator = nn.MSELoss(reduction='none')
        else:
            losscalculator = nn.MSELoss(reduction='mean')

        # Metrics tracking
        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        running_best_criterion = BestMeter("min")
        running_best_pr = BestMeter("max")
        running_best_train_loss = BestMeter("min")
        running_best_train_rmse = BestMeter("min")
        
        best_checkpoints = []
        no_improvement_count = 0
        best_score = float('-inf')
        best_epoch = 0
        
        # Training loop
        model.train()
        for epoch in range(args.epochs):
            pred_all = []
            label1_all = []
            label2_all = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch in pbar:
                graph1, graph2, extra, protein_values = batch
                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                extra = extra.to(device)
                protein_values = protein_values.to(device)
                pred = model(graph1, graph2, extra)
                label1 = graph1.y1
                label2 = graph1.y2

                label1_all.extend(label1.detach().cpu().numpy().flatten())
                label2_all.extend(label2.detach().cpu().numpy().flatten())
                pred_all.extend(pred.detach().cpu().numpy().flatten())
                
                # Calculate loss with specific system weighting if enabled
                if args.use_specific_fewshot:
                    protein_values_list = protein_values.tolist()
                    protein = ''.join(map(chr, [x for x in protein_values_list[0] if x != 0]))
                    if protein == args.specific_system or protein in args.specific_system:
                        loss = losscalculator(pred, label1) * 0.8  # Higher weight for specific system
                    else:
                        loss = losscalculator(pred, label1) * 0.2  # Lower weight for other systems
                else:
                    loss = losscalculator(pred, label1)
                
                # Apply AUE weighting if enabled
                if args.use_aue_weight:
                    if args.task_type == 'rbfe':
                        weights = (2 - label2) / 2
                    if args.task_type == 'abfe':
                        weights = (3 - label2) / 3
                    loss = (loss * weights).mean()
                
                # Add L2 regularization
                l2_reg = torch.tensor(0., requires_grad=True).to(device)
                for param in model.parameters():
                    l2_reg = l2_reg + torch.norm(param)
                loss = loss + 0.005 * l2_reg

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss.update(loss.item(), label1.size(0))
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(mean_squared_error(label1_all, pred_all))
            running_loss.reset()
      
            # Validation
            valid_rmse, valid_pr, valid_pred, valid_label, \
            valid_criterion = val(model, valid_loader, device)
            
            # Calculate normalized metrics for scoring
            norm_rmse = (10 - valid_rmse) / 10
            norm_pr = (valid_pr + 1) / 2
            norm_criterion = (10 - valid_criterion) / 10
            if args.app_mode == 'score':
                current_score = norm_criterion
            if args.app_mode == 'optimize':  #unabled for now （both score and optimize）
                current_score = norm_rmse
            
            # Update learning rate
            if epoch < 50:
                warmup_scheduler.step()
            else:
                scheduler.step(valid_rmse)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log training progress
            msg = '[Train] Epoch %05d | train_loss %.4f | train_rmse %.4f | lr %.6f' % (
                    epoch, epoch_loss, epoch_rmse, current_lr
            )
            logger.info(msg)
            msg = '[Validate] Epoch %05d | valid_rmse %.4f | valid_pr %.4f | valid_criterion %.4f | score %.4f' % (
                    epoch, valid_rmse, valid_pr, valid_criterion, current_score
            )
            logger.info(msg)

            # Save best model checkpoints
            improved = False
            if current_score > best_score:
                best_score = current_score
                improved = True

            model_name = f"epoch{epoch}-rmse{valid_rmse:.4f}-pr{valid_pr:.4f}-criterion{valid_criterion:.4f}-score{current_score:.4f}"

            if improved:
                best_epoch = epoch
                no_improvement_count = 0
                model_path = os.path.join(logger.get_model_dir(), model_name + '.pt')
                save_model_dict(model, logger.get_model_dir(), model_name)

                best_checkpoints.append((model_path, current_score))
                best_checkpoints.sort(key=lambda x: -x[1])
                if len(best_checkpoints) > args.max_checkpoints:
                    old_checkpoint = best_checkpoints.pop()
                    if os.path.exists(old_checkpoint[0]):
                        os.remove(old_checkpoint[0])
                logger.info(f'[Validate] Best score achieved: {current_score:.6f}')

            else:
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} epochs")
                
            # Early stopping
            if no_improvement_count >= args.early_stop_epoch:
                logger.info(f"Early stopping triggered. No improvement for {args.early_stop_epoch} epochs")
                logger.info(f"Best model was saved at epoch {best_epoch}")
                break
        
        # Final evaluation using best model
        best_checkpoint = best_checkpoints[0][0]
        load_model_dict(model, best_checkpoint)
        valid_rmse, valid_pr, valid_pred, valid_label, valid_criterion = val(model, valid_loader, device)

        logger.info(f"[Validate] Epoch %05d | valid_rmse1 %.6f | valid_pr1 %.6f | valid_criterion %.6f" % (
                    epoch, valid_rmse, valid_pr, valid_criterion)
                    )

# %%