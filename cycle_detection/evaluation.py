import torch
import numpy as np

@torch.no_grad()
def evaluate(model, loader, device=None, threshold: float = 0.3):
    """Evaluate model on dataset - returns accuracy, precision, recall, F1."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    y_true, y_pred, all_probs = [], [], []
    
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        # Probability of class 1 = cyclic
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()
        
        y_true.append(batch.y.view(-1).cpu())
        y_pred.append(preds.cpu())
        all_probs.append(probs.cpu())
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    all_probs = torch.cat(all_probs)

    # # === Diagnostics: show how threshold behaves ===
    # pos_rate = float((y_pred == 1).float().mean().item())
    # print(f"[eval] threshold={threshold:.2f}, predicted positive rate={pos_rate:.3f}")
    
    # mask_pos = (y_true == 1)
    # if mask_pos.any():
    #     p_pos = all_probs[mask_pos]
    #     between = int(((p_pos >= 0.30) & (p_pos < 0.50)).sum())
    #     below   = int((p_pos < 0.30).sum())
    #     print(f"[eval] positives in [0.30,0.50): {between}, below 0.30: {below}")

    # Overall accuracy
    accuracy = float((y_true.numpy() == y_pred.numpy()).mean())
    
    # Binary classification metrics (0=acyclic, 1=cyclic)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }