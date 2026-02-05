import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from pathlib import Path

class GraphBuilder:
    """Build PyTorch Geometric graph from processed user data"""
    
    def __init__(self, processed_data_path):
        """
        Args:
            processed_data_path: Path to users_processed.csv
        """
        self.data_path = Path(processed_data_path)
        self.users_df = None
        self.graph_data = None
    
    def load_data(self):
        """Load processed user data"""
        print("="*60)
        print("Loading Processed Data")
        print("="*60)
        
        self.users_df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.users_df)} users")
        print(f"  Features: {len(self.users_df.columns) - 2}")  # Exclude id and label
        
        return self.users_df
    
    def build_edges_knn(self, k=5):
        """
        Build edges using K-Nearest Neighbors based on feature similarity
        This creates connections between similar users
        
        Args:
            k: Number of nearest neighbors to connect
        """
        print("\n" + "="*60)
        print(f"Building Graph Edges (k-NN with k={k})")
        print("="*60)
        
        # Get feature columns (exclude id and label)
        feature_cols = [col for col in self.users_df.columns 
                       if col not in ['id', 'label']]
        
        # Extract features
        X = self.users_df[feature_cols].values
        
        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute pairwise distances (using subset for efficiency)
        print(f"Computing k-NN edges...")
        from sklearn.neighbors import NearestNeighbors
        
        # Use k+1 because first neighbor is the node itself
        knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        knn.fit(X_scaled)
        
        # Find k nearest neighbors for each node
        distances, indices = knn.kneighbors(X_scaled)
        
        # Build edge list
        edge_list = []
        for i in range(len(indices)):
            for j in indices[i][1:]:  # Skip first (self)
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"✓ Created {edge_index.shape[1]} directed edges")
        print(f"  Average degree: {edge_index.shape[1] / len(self.users_df):.2f}")
        
        return edge_index
    
    def build_edges_threshold(self, threshold=0.8, max_edges_per_node=10):
        """
        Build edges based on similarity threshold
        Connects nodes with high feature similarity
        
        Args:
            threshold: Similarity threshold (0-1)
            max_edges_per_node: Maximum edges per node to avoid dense graph
        """
        print("\n" + "="*60)
        print(f"Building Graph Edges (Similarity threshold={threshold})")
        print("="*60)
        
        # Get features
        feature_cols = [col for col in self.users_df.columns 
                       if col not in ['id', 'label']]
        X = self.users_df[feature_cols].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute cosine similarity (more efficient for large datasets)
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Computing pairwise similarities...")
        # Compute in batches to save memory
        batch_size = 1000
        edge_list = []
        
        for i in range(0, len(X_scaled), batch_size):
            batch_end = min(i + batch_size, len(X_scaled))
            batch = X_scaled[i:batch_end]
            
            similarities = cosine_similarity(batch, X_scaled)
            
            # For each node in batch, find top similar nodes
            for local_idx, sim_row in enumerate(similarities):
                global_idx = i + local_idx
                
                # Get indices above threshold
                similar_indices = np.where(sim_row > threshold)[0]
                similar_indices = similar_indices[similar_indices != global_idx]  # Remove self
                
                # Limit edges per node
                if len(similar_indices) > max_edges_per_node:
                    # Keep top-k most similar
                    top_k_indices = np.argsort(sim_row[similar_indices])[-max_edges_per_node:]
                    similar_indices = similar_indices[top_k_indices]
                
                # Add edges
                for j in similar_indices:
                    edge_list.append([global_idx, j])
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {batch_end}/{len(X_scaled)} nodes...")
        
        if len(edge_list) == 0:
            print("  Warning: No edges created with current threshold!")
            print("  Falling back to k-NN with k=5")
            return self.build_edges_knn(k=5)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"✓ Created {edge_index.shape[1]} directed edges")
        print(f"  Average degree: {edge_index.shape[1] / len(self.users_df):.2f}")
        
        return edge_index
    
    def create_pyg_data(self, edge_index, test_size=0.2, val_size=0.1):
        """
        Create PyTorch Geometric Data object
        
        Args:
            edge_index: Edge tensor
            test_size: Test split ratio
            val_size: Validation split ratio
        """
        print("\n" + "="*60)
        print("Creating PyTorch Geometric Data Object")
        print("="*60)
        
        # Get features (exclude id and label)
        feature_cols = [col for col in self.users_df.columns 
                       if col not in ['id', 'label']]
        
        # Node features
        x = torch.tensor(self.users_df[feature_cols].values, dtype=torch.float)
        
        # Node labels
        y = torch.tensor(self.users_df['label'].values, dtype=torch.long)
        
        print(f"Node features shape: {x.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        
        # Create train/val/test masks
        num_nodes = len(self.users_df)
        indices = np.arange(num_nodes)
        
        # Split: 70% train, 10% val, 20% test
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, 
            stratify=self.users_df['label']
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=42,
            stratify=self.users_df.loc[train_idx, 'label']
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"\nDataset splits:")
        print(f"  Train: {train_mask.sum()} nodes ({train_mask.sum()/num_nodes*100:.1f}%)")
        print(f"  Val: {val_mask.sum()} nodes ({val_mask.sum()/num_nodes*100:.1f}%)")
        print(f"  Test: {test_mask.sum()} nodes ({test_mask.sum()/num_nodes*100:.1f}%)")
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        self.graph_data = data
        
        print(f"\n✓ PyTorch Geometric Data object created")
        print(f"  {data}")
        
        return data
    
    def save_graph(self, output_path):
        """Save graph data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.graph_data, output_path)
        print(f"\n✓ Graph saved to {output_path}")
    
    def visualize_sample(self, num_nodes=100):
        """Visualize a small sample of the graph"""
        print("\n" + "="*60)
        print(f"Visualizing Sample Graph ({num_nodes} nodes)")
        print("="*60)
        
        # Sample nodes
        sample_indices = np.random.choice(len(self.users_df), num_nodes, replace=False)
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes with labels
        for idx in sample_indices:
            label = self.users_df.iloc[idx]['label']
            G.add_node(idx, label=label)
        
        # Add edges
        edge_index = self.graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in sample_indices and dst in sample_indices:
                G.add_edge(src, dst)
        
        print(f"Sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Visualization
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            # Color nodes by label
            colors = ['green' if G.nodes[node]['label'] == 0 else 'red' 
                     for node in G.nodes()]
            
            nx.draw(G, pos, node_color=colors, node_size=100, 
                   with_labels=False, edge_color='gray', alpha=0.6)
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Human'),
                Patch(facecolor='red', label='Bot')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("Sample Graph Visualization (Green=Human, Red=Bot)")
            
            # Save
            output_path = Path("data/processed/graph_visualization.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not create visualization: {e}")
