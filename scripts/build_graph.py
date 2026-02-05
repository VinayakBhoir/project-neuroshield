import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.graph_builder import GraphBuilder

def main():
    print("="*60)
    print("NeuroShield: Building Graph Structure")
    print("="*60)
    
    # Initialize graph builder
    data_path = "data/processed/users_processed.csv"
    builder = GraphBuilder(data_path)
    
    # Load data
    builder.load_data()
    
    # Build edges using k-NN (faster, good for starting)
    # Option 1: k-NN approach
    edge_index = builder.build_edges_knn(k=10)
    
    # Option 2: Similarity threshold (uncomment to try)
    # edge_index = builder.build_edges_threshold(threshold=0.7, max_edges_per_node=15)
    
    # Create PyTorch Geometric data
    data = builder.create_pyg_data(edge_index)
    
    # Save graph
    builder.save_graph("data/processed/graph_data.pt")
    
    # Visualize sample
    builder.visualize_sample(num_nodes=200)
    
    print("\n" + "="*60)
    print("✓ Graph building complete!")
    print("="*60)
    print("\nNext step: Train GNN model on this graph")

if __name__ == "__main__":
    main()
