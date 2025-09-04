import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.utils import class_weight
import numpy as np
import copy
import pandas as pd
from sklearn.metrics import roc_auc_score

class TorchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            self.labels = torch.tensor(labels.values, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# Hàm get_class_balanced_weights không thay đổi
def get_class_balanced_weights(y_train, beta=0.999):
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    unique, counts = np.unique(y_train, return_counts=True)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(unique)
    return torch.tensor(weights, dtype=torch.float32)

def train_pytorch_model(model, X_train, y_train, X_val, y_val, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(dataset=TorchDataset(X_train, y_train), batch_size=cfg.TRAINING_PARAMS['batch_size'], shuffle=True)

    label_smoothing = cfg.TRAINING_PARAMS.get('label_smoothing', 0.0)
    optimizer_name = cfg.TRAINING_PARAMS.get('optimizer', 'Adam').lower()
    use_scheduler = cfg.TRAINING_PARAMS.get('use_scheduler', False)

    unique_classes = np.unique(y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train)
    weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=(y_train.values.ravel() if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
    print(f"   -> Sử dụng Cross Entropy Loss (label_smoothing={label_smoothing}).")

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING_PARAMS['lr'], weight_decay=cfg.TRAINING_PARAMS.get('weight_decay', 0))
        print("   -> Sử dụng Optimizer: AdamW")
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING_PARAMS['lr'], weight_decay=cfg.TRAINING_PARAMS.get('weight_decay', 0))
        print("   -> Sử dụng Optimizer: Adam")

    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAINING_PARAMS['epochs'], eta_min=1e-7)
        print("   -> Sử dụng Learning Rate Scheduler: CosineAnnealingLR")

    best_val_loss = float('inf'); patience_counter = 0; best_model_state = None


    history = []

    y_val_numpy = y_val.values if isinstance(y_val, (pd.Series, pd.DataFrame)) else y_val
    y_val_tensor = torch.tensor(y_val_numpy, dtype=torch.long).to(device)
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    for epoch in range(cfg.TRAINING_PARAMS['epochs']):
        model.train()
        total_train_loss = 0
        train_targets, train_preds = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_train_loss += loss.item()
            train_targets.extend(labels.cpu().numpy())
            train_preds.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())

        if use_scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_preds_proba = torch.softmax(val_outputs, dim=1).cpu().numpy()

        avg_train_loss = total_train_loss / len(train_loader)
        train_auc = roc_auc_score(train_targets, train_preds, multi_class='ovr', average='macro')
        val_auc = roc_auc_score(y_val_numpy, val_preds_proba, multi_class='ovr', average='macro')

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'train_auc': train_auc,
            'val_auc': val_auc
        })

        if (epoch + 1) % 10 == 0:
            lr_info = f", LR: {scheduler.get_last_lr()[0]:.8f}" if use_scheduler else ""
            print(f"     Epoch [{epoch+1:03d}/{cfg.TRAINING_PARAMS['epochs']}], Val Loss: {val_loss:.6f}{lr_info}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= cfg.TRAINING_PARAMS.get('early_stopping_patience', 15):
            print(f"   -> Early stopping tại epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history
