import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

class BotDetectionTrainer:
    """Training pipeline for bot detection GNN"""
    
    def __init__(self, model, data, learning_rate=0.001, weight_decay=5e-4):
        self.model = model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model and data to device
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # For tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"✓ Trainer initialized on device: {self.device}")
    
    def train_epoch(self):
        """Single training epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss only on training nodes
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        pred = out[self.data.train_mask].argmax(dim=1)
        correct = pred == self.data.y[self.data.train_mask]
        acc = correct.sum().item() / self.data.train_mask.sum().item()
        
        return loss.item(), acc
    
    @torch.no_grad()
    def validate(self):
        """Validation step"""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        
        # Loss on validation set
        loss = F.nll_loss(out[self.data.val_mask], self.data.y[self.data.val_mask])
        
        # Accuracy
        pred = out[self.data.val_mask].argmax(dim=1)
        correct = pred == self.data.y[self.data.val_mask]
        acc = correct.sum().item() / self.data.val_mask.sum().item()
        
        return loss.item(), acc
    
    def train(self, num_epochs=100, early_stopping_patience=10, verbose=True):
        """Full training loop with early stopping"""
        print("\n" + "="*60)
        print(f"Starting Training for {num_epochs} epochs")
        print("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✓ Restored best model (Val Loss: {best_val_loss:.4f})")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training complete in {elapsed_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': best_val_loss
        }
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)
        print(f"✓ Checkpoint saved to {path}")
