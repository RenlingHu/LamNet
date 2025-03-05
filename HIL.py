import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn

class HIL(MessagePassing):
    """
    Heterogeneous Interaction Layer for molecular graphs
    
    This layer processes both intra-molecular and inter-molecular interactions
    using message passing and coordinate information.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MLP for processing covalent interactions
        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
            
        # MLP for processing non-covalent interactions
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        # MLPs for processing coordinate information
        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        """
        Forward pass of the layer
        
        Args:
            x (Tensor): Node feature matrix
            edge_index_intra (Tensor): Edge indices for intra-molecular bonds
            edge_index_inter (Tensor): Edge indices for inter-molecular interactions
            pos (Tensor, optional): Node position coordinates
            size (tuple, optional): Size of the graph
            
        Returns:
            Tensor: Updated node features
        """
        # Process intra-molecular interactions
        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]

        # Calculate radial basis functions for distances
        radial_cov = self.mlp_coord_cov(_rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        # Process inter-molecular interactions if they exist
        if edge_index_inter is not None:
            row_ncov, col_ncov = edge_index_inter
            coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
            radial_ncov = self.mlp_coord_ncov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
            out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)
            # Combine intra and inter molecular features
            out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)
        else:
            out_node = self.mlp_node_cov(x + out_node_intra)

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial,
                index: Tensor):
        """
        Message function for the message passing scheme
        
        Args:
            x_j (Tensor): Source node features
            x_i (Tensor): Target node features  
            radial (Tensor): Radial basis function values
            index (Tensor): Index tensor
            
        Returns:
            Tensor: Message features
        """
        x = x_j * radial
        return x


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    Radial basis function expansion
    
    Converts distances into RBF features using Gaussian functions
    
    Args:
        D (Tensor): Distance tensor
        D_min (float): Minimum distance for RBF
        D_max (float): Maximum distance for RBF  
        D_count (int): Number of RBF functions
        device (str): Device to place tensors on
        
    Returns:
        Tensor: RBF features with shape [...dims, D_count]
    """
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# %%