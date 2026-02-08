import json
from jsonschema import validate, ValidationError
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import data_model

# Init Test
def init(path: Path) -> dict:
    
    # Json
    with open(Path(__file__).parent / "config_test.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Import Json schema
    with open(Path(__file__).parent / "config_test.schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    # validate
    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise SystemExit(f"[CONFIG ERROR] {e.message}")

    # Seed
    data_model.set_seed(cfg["seed"])
    
    # Device
    device = data_model.pick_device(cfg.get("device", "auto"))
    
    return cfg, device
      
# Labels
def load_labels(out_dir: Path):
    labels_path = out_dir / "labels.json"
    
    # Check file exists
    if not labels_path.exists():
        return None
    
    # Load class
    with open(labels_path, "r", encoding="utf-8") as f:
        class_test = json.load(f)
        
    
    return [class_test[str(i)] for i in range(len(class_test))]

# Evaluate best model
def evaluate(model, loader, criterion, device):
    
    # Evaluation mode
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    # No gradients
    with torch.no_grad():
        # Batch loop
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Model prediction
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Batch size
            batch_size = labels.size(0)
            
            # Update metrics
            loss_sum += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    # Final metrics
    avg_loss = loss_sum / max(1, len(total))
    acc = correct / max(1, total)
    
    return avg_loss, acc, total

# Preparation test
def build_test(cfg: dict, test_dir: Path, out_dir: Path, device):
    
    # Dataset and dataloader
    test_dataset  = ImageFolder(root=str(test_dir), transform=data_model.get_val_transform(cfg["img_size"]))
    test_loader  = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, pin_memory=(device.type == "cuda"))

    # Load training labels
    train_classes = load_labels(out_dir)
    if train_classes is not None:
        
        # Check class order
        if train_classes != list(test_dataset.classes):
            raise ValueError("Ordine classi diverso tra training(labels.json) e test(ImageFolder)")
        # Classes from training
        num_classes = len(train_classes)
    
    else:
        # Classes from test
        num_classes = len(test_dataset.classes)


    return test_dataset, test_loader, num_classes

# Test best model
def test(cfg: dict, device, test_dir: Path, out_dir: Path) -> dict:
    
    # Build test setup
    test_dataset, test_loader, num_classes = build_test(cfg, test_dir, out_dir, device)

    # Build model
    model = data_model.build_model(num_classes=num_classes, freeze_backbone=cfg.get("freeze_backbone", False), pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout_p", 0.5),).to(device)
    
    # Load best model
    best_path = out_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best_model.pt non trovato in: {best_path}")
    
    # Load best weights
    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, n = evaluate(model, test_loader, criterion, device)

    # Build results
    results = {"test_dir": str(test_dir), "out_dir": str(out_dir), "best_model_path": str(best_path), "num_samples": int(n), "test_loss": float(test_loss), "test_accuracy": float(test_acc), "device": str(device), "classes": list(test_dataset.classes),}

    # Save result
    results_path = out_dir / "test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results



def main():
    
    cfg, device = init()

    # Path
    test_dir = Path(cfg["test_dir"])
    out_dir = Path(cfg["out_dir"])

    # check path
    if not test_dir.exists():
        raise FileNotFoundError(f"Cartella test_dir non trovata: {test_dir}")
    if not out_dir.exists():
        raise FileNotFoundError(f"Cartella out_dir non trovata: {out_dir}")

    # Run test
    results = test(cfg, device, test_dir, out_dir)
    print(f"Test Results\nLoss: {results['test_loss']:.4f}\nAccuracy: {results['test_accuracy']:.4f}\nSamples: {results['num_samples']}\nModel: {results['best_model_path']}"
)


if __name__ == "__main__":
    main()