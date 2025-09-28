import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from eval import evaluate_with_baselines

def _bce_with_pos_weight(pos_weight: float, device):
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

def train_loop(model, train_loader: DataLoader, val_loader: DataLoader, device,
               epochs: int = 20, lr: float = 1e-3, wd: float = 1e-5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            # print sample dim
            if running == -10.0:
                print(logits[:20])
                print(logits.shape)

            pos = float(batch.edge_y.sum().item())
            neg = float(batch.edge_y.numel() - batch.edge_y.sum().item())
            pos_weight = (neg / max(pos, 1.0)) if (pos + neg) > 0 else 1.0
            loss_fn = _bce_with_pos_weight(pos_weight, device)
            loss = loss_fn(logits, batch.edge_y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

        val = evaluate_with_baselines(model, val_loader, device)
        if val["approx_ratio"] > best_val:
            best_val = val["approx_ratio"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train_loss={running/len(train_loader):.4f} "
              f"| val_edge_acc={val['edge_acc']:.3f} | val_auroc={val['auroc']:.3f} | val_auprc={val['auprc']:.3f} "
              f"| val_approx={val['approx_ratio']:.3f} (rand={val['approx_rand']:.3f}, deg={val['approx_deg']:.3f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model