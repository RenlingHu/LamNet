# LamNet

LamNet: An Alchemical-Path-Aware Graph Neural Network for Accelerating Relative and Absolute Binding Free Energy Calculations

![LamNet Overview](Figure1.png)

## Project Structure

### Core Components
- `LamNet.py`: Core implementation of the LamNet model architecture
- `HIL.py`: Hierarchical Interaction Learning module for molecular interactions
- `dataset.py`: Comprehensive dataset processing and loading utilities
- `utils.py`: General utility functions and helper tools
- `logger.py`: Logging and experiment tracking utilities

### Data Processing
- `00.preprocessing_abfe.py`: Preprocessing pipeline for absolute binding free energy calculations
- `00.preprocessing_rbfe.py`: Preprocessing pipeline for relative binding free energy calculations

### Model Development
- `01.1_train.py`: Main training script with multiple training modes
- `01.2_generalize.py`: Model generalization and testing framework

### Analysis Tools
- `02.1_score.py`: Unified scoring script for RBFE and ABFE predictions
- `02.2_optimize.py`: Advanced alchemical parameter optimization tools

### Directory Structure
- `model/`: Model checkpoints and saved states
- `data/`: Training and evaluation datasets
- `score/`: Scoring workspace and results
- `optimize/`: Parameter optimization workspace
- `multi/`: Multi-target/host model training
- `fewshot/`: Few-shot learning experiments
- `single/`: Target-specific model training

## Requirements
For detailed dependency information, please refer to `requirements.txt`

## Usage Guide

### 1. Data Preprocessing
```bash
# RBFE
# training data preprocessing
python 00.preprocessing_abfe.py --mode=train --csv_name=datasets
# scoring data preprocessing
python 00.preprocessing_abfe.py --mode=score --csv_name=CB7

# ABFE
# training data preprocessing
python 00.preprocessing_rbfe.py --mode=train --csv_name=datasets --input_ligand_format=mol2
# scoring data preprocessing
python 00.preprocessing_rbfe.py --mode=score --csv_name=CDK2-weak --input_ligand_format=sdf
# optimizing data preprocessing
python 00.preprocessing_rbfe.py --mode=optimize --csv_name=CDK2_1oiu_1h1q --input_ligand_format=sdf
```

### 2. Model Training
```bash
# RBFE
# all-target training
python 01.1_train.py --task_type=rbfe --mode=multi --system=all --use_aue_weight=True
# leave-one-out training
python 01.1_train.py --task_type=rbfe --mode=multi --system=CDK2 --use_aue_weight=True
# specific-target training
python 01.1_train.py --task_type=rbfe --mode=single --system=CDK2 --use_aue_weight=True --batch_size=16
# few-shot training
python 01.1_train.py --task_type=rbfe --mode=fewshot --system=PTP1B --use_aue_weight=True --use_sepcific_fewshot=False

# ABFE
# all-host training
python 01.1_train.py --task_type=abfe --mode=multi --system=all --use_aue_weight=False


# Test model generalization
# RBFE
python 01.2_generalize.py --model_path='LamNet/model/rbfe/rbfe_pl/CDK2_w/model/epoch98-rmse4.8701-pr0.9762-criterion6.7655-score0.3234.pt'
```

### 3. Model Evaluation and Optimization
```bash
# Binding free energy scoring
# RBFE
python 02.1_score.py --task_type=rbfe --system=CDK2 --connect=weak --model_path='LamNet/model/rbfe/rbfe_pl/CDK2_w/model/epoch98-rmse4.8701-pr0.9762-criterion6.7655-score0.3234.pt'
# ABFE
python 02.1_score.py --task_type=abfe --system=CB7 --model_path='LamNet/model/abfe/abfe_gh/model/epoch45-rmse21.9192-pr0.6414-criterion9.4640-score0.0536.pt'

# Alchemical parameter optimization
# RBFE
python 02.2_optimize.py --cutoff=10 --system=CDK2_1oiu_1h1q --model_path='LamNet/model/rbfe/rbfe_pl/CDK2_w/model/epoch98-rmse4.8701-pr0.9762-criterion6.7655-score0.3234.pt'
```

## License
This project is licensed under the open-source license. See the `LICENSE` file for details.

## Citation
If you use LamNet in your research, please cite our work.
[unpublished]

## Contact
For any questions or issues, please contact us through the Issue system.
