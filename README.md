# LamNet

LamNet: an Alchemical-Path-Aware Graph Neural Network to Accelerate Binding Free Energy Calculations for Drug Discovery and Beyond

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

We provide **part of** our training data (two targets for RBFE and two hosts for ABFE) to help readers understand the network architecture and training process of LamNet.

We also provide **all of** our valid, test set to allow readers to reproduce the results metioned in our article using our trained model checkpoint.

### 1. Data Preprocessing (already done)
```bash
# RBFE
# training data preprocessing 
python 00.preprocessing_rbfe.py --mode=train --csv_name=datasets --input_ligand_format=sdf
# scoring data preprocessing
python 00.preprocessing_rbfe.py --mode=score --csv_name=CDK2-weak --input_ligand_format=sdf
# optimizing data preprocessing
python 00.preprocessing_rbfe.py --mode=optimize --csv_name=CDK2_1oiu_1h1q --input_ligand_format=sdf

# ABFE
# training data preprocessing
python 00.preprocessing_abfe.py --mode=train --csv_name=datasets
# scoring data preprocessing
python 00.preprocessing_abfe.py --mode=score --csv_name=CB7
```

### 2. Model Training

The example of **RBFE all-target training** using publicly available datasets can be found in the `model/`.

The checkpoints used for all the experiments in the article are provided under the corresponding categories in the `model/`.

```bash
# RBFE
# all-target training
python 01.1_train.py --task_type=rbfe --mode=multi --system=all --use_aue_weight=True
# leave-one-out training
python 01.1_train.py --task_type=rbfe --mode=multi --system=BACE --use_aue_weight=True ##--batch_size=32
# specific-target training
python 01.1_train.py --task_type=rbfe --mode=single --system=BACE --use_aue_weight=True ##--batch_size=16
# few-shot training 
python 01.1_train.py --task_type=rbfe --mode=fewshot --system=BACE --use_aue_weight=True

# ABFE
# all-host training
python 01.1_train.py --task_type=abfe --mode=multi --system=all --use_aue_weight=False

# Test model generalization
# RBFE
python 01.2_generalize.py --model_path='model/rbfe/rbfe_targetspecific/BACE_w/model/epoch100-rmse2.1705-pr0.9950-criterion1.8158-score0.8184.pt'
```

### 3. Model Evaluation and Optimization

The checkpoints used for all the experiments in the article are provided under the corresponding categories in the `model/`.

```bash
# Binding free energy scoring
# RBFE
python 02.1_score.py --task_type=rbfe --system=BACE --connect=weak --model_path='model/rbfe/rbfe_pl/BACE_w/model/epoch59-rmse4.9747-pr0.9758-criterion7.4391-score0.2561.pt'
# ABFE
python 02.1_score.py --task_type=abfe --system=CB7 --model_path='model/abfe/abfe_gh/model/epoch45-rmse21.9192-pr0.6414-criterion9.4640-score0.0536.pt'

# Alchemical parameter optimization
# RBFE
python 02.2_optimize.py --cutoff=10 --system=CDK2_1oiu_1h1q --model_path='model/rbfe/rbfe_pl/CDK2_w/model/epoch98-rmse4.8701-pr0.9762-criterion6.7655-score0.3234.pt'
```

## License
This project is licensed under the open-source license. See the `LICENSE` file for details.

## Citation
If you use LamNet in your research, please cite our work.
[unpublished]

## Contact
For any questions or issues, please contact us through the Issue system.
