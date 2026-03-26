import numpy as np
from utils import get_device, load_config, set_seed
from models import load_model
from transforms import build_transform
from pathlib import Path
from datasets import build_dataset, build_dataloaders, build_splits
from engine import evaluate
from metrics import compute_classification_metrics, plot_confusion_matrix

def main():

    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_DIR = BASE_DIR / "configs" / "config.yaml"
    IMAGES_DIR = BASE_DIR / "data" / "NEU-DET" / "IMAGES"
    ANNOTATIONS_DIR = BASE_DIR / "data" / "NEU-DET" / "ANNOTATIONS"
    MODEL_DIR = BASE_DIR / "models"
    BEST_MODEL = MODEL_DIR / "best_model.pth"

    config = load_config(CONFIG_DIR)
    set_seed(config["training"]["seed"])
    device = get_device()

    model = load_model(checkpoint_path=BEST_MODEL,
                       num_classes=config["training"]["num_classes"])
    
    transform = build_transform()
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

    model.to(device)

    targets, preds = evaluate(model=model,
                        dataloader=test_loader,
                        device=device
                        )
    metrics = compute_classification_metrics(y_true=targets, y_pred=preds)

    print(metrics)

    plot_confusion_matrix(cm=metrics["confusion_matrix"], class_names=dataset.classes)

if __name__ == "__main__":
    main()
    