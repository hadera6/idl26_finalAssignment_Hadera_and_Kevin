"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import copy
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device) 
                labels = labels.squeeze().long().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())  

        acc  = (correct / total) * 100
        loss = running_loss / total
        
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec  = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
        f1   = f1_score(all_labels,       all_preds, average='macro', zero_division=0)

        return loss, acc, prec, rec, f1

    def fit(self, train_loader, val_loader, epochs,
            scheduler=None,
            patience=10,                                                        # E.1
            no_restore=False
        ):
        print("\n Starting Training Routine...")
        print("-" * 50)

        best_val_loss     = float('inf')                                       # E.2
        best_weights      = None                                               # E.2
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, *_ = self.evaluate(val_loader)

            if scheduler is not None:                                          # D.3
                scheduler.step()
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:                                       # E.3
                best_val_loss     = val_loss
                best_weights      = copy.deepcopy(self.model.state_dict())    # E.3
                epochs_no_improve = 0
                print(f"  Best model saved at epoch {epoch+1}")
            else:
                epochs_no_improve += 1                                         # E.4
                if epochs_no_improve >= patience:                              # E.4
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"— no improvement for {patience} epochs")
                    break                                                      # E.4

        if best_weights is not None and not no_restore:                        # F.2
            self.model.load_state_dict(best_weights)                          # E.5
            print(f"  Best weights restored — val loss {best_val_loss:.4f}")
        elif no_restore:                                                       # F.3
            print(f"  no_restore=True — using final epoch weights")
        
        print("-" * 50)
        print("Training Complete!")