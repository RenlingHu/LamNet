import os
import pickle
import torch

def normalize(x):
    """Normalize input tensor to [0,1] range"""
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    """Create directories if they don't exist
    
    Args:
        dir_list: List of directory paths to create
    """
    assert isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    """Save model state dict to file
    
    Args:
        model: Model to save
        model_dir: Directory to save model
        msg: Message to append to filename
    """
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("Model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    """Load model state dict from checkpoint file
    
    Args:
        model: Model to load weights into
        ckpt: Path to checkpoint file
    """
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    """Recursively delete all files in directory
    
    Args:
        path: Directory path to delete files from
    """
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    """Write object to pickle file
    
    Args:
        filename: Output pickle filename
        obj: Python object to pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    """Read object from pickle file
    
    Args:
        filename: Input pickle filename
    Returns:
        Unpickled Python object
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class BestMeter(object):
    """Tracks best value seen during training
    
    Args:
        best_type: 'min' or 'max' to track minimum or maximum value
    """
    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        """Reset best value tracker"""
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        """Update best value
        
        Args:
            best: New best value
        """
        self.best = best
        self.count = 0

    def get_best(self):
        """Get current best value"""
        return self.best

    def counter(self):
        """Increment counter and return value"""
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores running average of values"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset average meter"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update running average with new value
        
        Args:
            val: New value
            n: Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        """Calculate and return current average"""
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg
