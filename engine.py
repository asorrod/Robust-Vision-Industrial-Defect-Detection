import torch
from tqdm.auto import tqdm

def train_step(model:torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device) -> tuple[float, float]:
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def validate_step(model:torch.nn.Module,
              dataloader: torch.utils.data.dataloader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> tuple[float, float]:
    
    eval_loss, eval_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            eval_pred_logits = model(X)
            loss = loss_fn(eval_pred_logits, y)
            eval_loss += loss.item()
            eval_pred_labels = eval_pred_logits.argmax(dim=1)
            eval_acc += ((eval_pred_labels == y).sum().item()/len(eval_pred_labels))
        
        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)

    return eval_loss, eval_acc

def build_optimizer(model: torch.nn.Module,
                    lr: float):
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=lr)
    return optimizer

def build_loss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn

def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device):

    model.eval()
    
    all_preds = []
    all_targets= []

    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            
            y_pred_logits = model(X)
            y_preds = torch.argmax(y_pred_logits, dim=1)

            all_preds.extend(y_preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    return all_targets, all_preds