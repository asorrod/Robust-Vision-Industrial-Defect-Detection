import torch
from pathlib import Path
from utils import get_device, load_config, set_seed
from datasets import build_dataset, build_splits, build_dataloaders
from tqdm.auto import tqdm
from engine import train_step, validate_step, build_optimizer, build_loss
from transforms import build_base_transform
from models import build_model

def main():

    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_DIR = BASE_DIR / "configs" / "config.yaml"
    IMAGES_DIR = BASE_DIR / "data" / "NEU-DET" / "IMAGES"
    ANNOTATIONS_DIR = BASE_DIR / "data" / "NEU-DET" / "ANNOTATIONS"
    MODEL_DIR = BASE_DIR / "models"

    config = load_config(CONFIG_DIR)
    set_seed(config["training"]["seed"])
    device = get_device()
    
    model = build_model(config["training"]["num_classes"]).to(device)

    transform = build_base_transform()
    dataset = build_dataset(images_dir=IMAGES_DIR,
                             annotations_dir=ANNOTATIONS_DIR,
                             transform=transform)

    train_dataset, val_dataset, test_dataset = build_splits(dataset=dataset,
                                                            train_ratio=config["training"]["train_ratio"],
                                                            val_ratio=config["training"]["val_ratio"])

    
    train_loader, val_loader, test_loader = build_dataloaders(train_dataset=train_dataset,
                                                              val_dataset=val_dataset,
                                                              test_dataset=test_dataset,
                                                              batch_size=config["training"]["batch_size"],
                                                              num_workers=config["data"]["num_workers"],
                                                              pin_memory=config["data"]["pin_memory"])

    optimizer = build_optimizer(model=model,
                                lr=config["training"]["lr"])
    loss = build_loss()

    best_val_loss= float("inf")
    patience = config["training"]["patience"]
    counter = 0

    for epoch in tqdm(range(config["training"]["epochs"])):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_loader,
                                           loss_fn=loss,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = validate_step(model=model,
                                          dataloader=val_loader,
                                          loss_fn=loss,
                                          device=device)
        
        tqdm.write(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            tqdm.write(f"New best acc - Saving model...")
            
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, MODEL_DIR / "best_model.pth")
        else: 
            counter += 1
            tqdm.write(f"Epoch {epoch+1}: No improvement. EarlyStopping counter: {counter}/{patience}")

        if counter >= patience:
            tqdm.write(f"--- EARLY STOPPING in epoch {epoch+1} ---")
            break

    return None

if __name__ == "__main__":
    main()