import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn_models import GCNBot, GraphSAGEBot, GATBot
from training.trainer import BotDetectionTrainer
from training.evaluator import ModelEvaluator

def main():
    print("="*60)
    print("NeuroShield: Training GNN Model")
    print("="*60)
    
    # Load graph data (PyTorch 2.6+ compatibility)
    print("\nLoading graph data...")
    data = torch.load("data/processed/graph_data.pt", weights_only=False)
    print(f"✓ Loaded graph: {data}")
    
    # Initialize model
    print("\nInitializing model...")
    num_features = data.x.shape[1]
    num_classes = 2  # Binary: Human vs Bot
    
    # Choose model type
    model = GCNBot(
        num_features=num_features,
        hidden_channels=64,
        num_classes=num_classes,
        num_layers=3,
        dropout=0.5
    )
    
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = BotDetectionTrainer(
        model=model,
        data=data,
        learning_rate=0.001,
        weight_decay=5e-4
    )
    
    # Train
    history = trainer.train(
        num_epochs=200,
        early_stopping_patience=20,
        verbose=True
    )
    
    # Evaluate on all splits
    evaluator = ModelEvaluator(model, data)
    
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    train_metrics = evaluator.print_report('train')
    val_metrics = evaluator.print_report('val')
    test_metrics = evaluator.print_report('test')
    
    # Save model
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    model_path = Path("saved_models/neuroshield_gcn.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save training history
    import json
    history_path = Path("saved_models/training_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items() if isinstance(vals, list)}, f)
    print(f"✓ Training history saved to {history_path}")
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
