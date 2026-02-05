import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Linear, Dropout

class GCNBot(torch.nn.Module):
    """Graph Convolutional Network for bot detection"""
    
    def __init__(self, num_features, hidden_channels=64, num_classes=2, num_layers=3, dropout=0.5):
        super(GCNBot, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.conv1 = GCNConv(num_features, hidden_channels)
        
        # Hidden layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.conv_out = GCNConv(hidden_channels, hidden_channels)
        
        # Classifier
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

class GraphSAGEBot(torch.nn.Module):
    """GraphSAGE model for bot detection"""
    
    def __init__(self, num_features, hidden_channels=64, num_classes=2, num_layers=3, dropout=0.5):
        super(GraphSAGEBot, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.conv1 = SAGEConv(num_features, hidden_channels)
        
        # Hidden layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.conv_out = SAGEConv(hidden_channels, hidden_channels)
        
        # Classifier
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

class GATBot(torch.nn.Module):
    """Graph Attention Network for bot detection with explainability"""
    
    def __init__(self, num_features, hidden_channels=64, num_classes=2, num_layers=3, dropout=0.5, heads=8):
        super(GATBot, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Input layer (multi-head attention)
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        
        # Hidden layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        
        # Output layer (single head)
        self.conv_out = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        
        # Classifier
        self.lin = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, return_attention_weights=False):
        # First layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Hidden layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        # Output layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            x, attention_weights = self.conv_out(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv_out(x, edge_index)
        
        # Classification
        x = self.lin(x)
        
        if return_attention_weights:
            return F.log_softmax(x, dim=1), attention_weights
        else:
            return F.log_softmax(x, dim=1)
