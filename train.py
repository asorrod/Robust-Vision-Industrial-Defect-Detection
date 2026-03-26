import torch
from pathlib import Path
from utils import get_device, load_config, set_seed
from datasets import build_dataset, build_splits, build_dataloaders, build_splits_index
from tqdm.auto import tqdm
from engine import train_step, validate_step, build_optimizer, build_loss
from transforms import build_transform
from models import build_model
from transforms import build_corruption_transforms
from torchvision.transforms import v2
from torch.utils.data import Subset

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

    data_augment = config["robustness_training"]["data_augment"]
    
    model_name = "robust_model.pth" if data_augment else "baseline_model.pth"

    perturbations = []

    if data_augment:

        corruption_names = config["robustness_training"]["corruptions"]
        severity = config["robustness_training"]["severity_levels"]
        prob = config["robustness_training"]["probability"]

        corruption_transforms = [
            build_corruption_transforms(name, severity)
            for name in corruption_names
        ]

        perturbations.append(
            v2.RandomApply(
                [v2.RandomChoice(corruption_transforms)],
                p=prob
            )
        )

    base_dataset = build_dataset(
    images_dir=IMAGES_DIR,
    annotations_dir=ANNOTATIONS_DIR,
    transform=None
    )

    # Obtener índices de split
    train_idx, val_idx, test_idx = build_splits_index(
        dataset=base_dataset,
        train_ratio=config["training"]["train_ratio"],
        val_ratio=config["training"]["val_ratio"]
    )

    # Transforms
    train_transform = build_transform(perturbations)
    eval_transform = build_transform([])

    # Crear datasets independientes
    train_dataset = Subset(
        build_dataset(IMAGES_DIR, ANNOTATIONS_DIR, train_transform),
        train_idx
    )

    val_dataset = Subset(
        build_dataset(IMAGES_DIR, ANNOTATIONS_DIR, eval_transform),
        val_idx
    )

    test_dataset = Subset(
        build_dataset(IMAGES_DIR, ANNOTATIONS_DIR, eval_transform),
        test_idx
    )
    
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
            }, MODEL_DIR / model_name)
        else: 
            counter += 1
            tqdm.write(f"Epoch {epoch+1}: No improvement. EarlyStopping counter: {counter}/{patience}")

        if counter >= patience:
            tqdm.write(f"--- EARLY STOPPING in epoch {epoch+1} ---")
            break

    return None

if __name__ == "__main__":
    main()