from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import numpy as np

class ModelEvaluator:
    """Evaluate trained model"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate on given mask (train/val/test)"""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        pred = out[mask].argmax(dim=1)
        
        y_true = self.data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        return y_true, y_pred
    
    def get_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }
        return metrics
    
    def print_report(self, split='test'):
        """Print detailed evaluation report"""
        print("\n" + "="*60)
        print(f"Evaluation Report ({split.upper()} SET)")
        print("="*60)
        
        # Get appropriate mask
        if split == 'train':
            mask = self.data.train_mask
        elif split == 'val':
            mask = self.data.val_mask
        else:
            mask = self.data.test_mask
        
        # Evaluate
        y_true, y_pred = self.evaluate(mask)
        
        # Metrics
        metrics = self.get_metrics(y_true, y_pred)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Human    Bot")
        print(f"Actual Human  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       Bot    {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Human', 'Bot']))
        
        return metrics
