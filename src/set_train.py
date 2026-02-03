from pathlib import Path
import json
import torch

# Create dir result for train
def init_set(result_dir: str) -> Path:
    # Create result directory 
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints directory
    checkpoints_dir = Path(result_dir/"checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir

# Save result train
def save_last_model(model, result_dir: Path) -> None:
    
    # Create save model
    last_model_path = Path(result_dir/"last_model.pt")
    torch.save(model.state_dict(), last_model_path)
    
# Save best model
def save_best_model(model, result_dir: Path) -> None:
    
    # Create save best model
    best_model_path = Path(result_dir/"best_model.pt")
    torch.save(model.state_dict(), best_model_path)
    
# Checkpoint of weight
def save_checkpoint(model, optimizer, result_dir: Path, epoch: int, best_metric: float) -> None:
    
    # Create result checkpoint for this epoch
    checkpoint_path = result_dir / "checkpoints" / f"epoch_{epoch}.pt"
    torch.save({"epoch": epoch, "best_metric": best_metric, "best_metric": model.state_dict(), "optimizer_state": optimizer.state_dict()}, checkpoint_path)
         
# Save history
def save_history(history: dict, result_dir: Path) -> None:
    
    # Create history
    history_path = Path(result_dir/"history.json")
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
              
# Update history with new value 
def update_history(history: dict, train_loss: float, val_loss: float, val_acc: float, epoch_time: float) -> dict:
    if not history:
        history["train_loss"] = []
        history["val_loss"] = []
        history["val_acc"] = []
        history["epoch_time"] = []
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["epoch_time"].append(epoch_time)
        
    return history
          
#Early stop of train
def early_stop(val_acc: float, best_val_acc: float, bad_epochs: int, patience: int, early_stopping: bool, target_accuracy: float = None) -> bool:
    
    # Arrived at the accuracy target
    if target_accuracy is not None and val_acc >= target_accuracy:
        return True

    # Early stop deactivated
    if early_stopping is False:
        return False
    
    # Patience
    if bad_epochs >= patience:
        return True
    else:
        return False

#Tensorboard
def tensorboard(writer, epoch: int, train_loss: float, val_loss: float, val_acc: float, epoch_time: float) -> None:
    if writer is None:
        return
    
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Time/epoch_sec", epoch_time, epoch)
    
    
    