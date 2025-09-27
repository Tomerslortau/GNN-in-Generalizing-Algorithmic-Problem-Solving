import torch
import torch.nn as nn
from evaluation import evaluate

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Run one training epoch and return average loss."""
    model.train()
    total_loss, total_graphs = 0.0, 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    
    return total_loss / max(1, total_graphs)

def train(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-4, device=None):
    """Train model with validation monitoring."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    print(f"Training for {epochs} epochs...")
    print("Progress updates every 10 epochs:")
    print("-" * 50)
    
    for epoch in range(1, epochs + 1):
        # Train one epoch
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        model.eval()
        with torch.no_grad():
            metrics = evaluate(model, val_loader, device)
        val_acc = metrics['accuracy']
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"  -> New best validation accuracy: {val_acc:.4f}")
    
    print("-" * 50)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model