import json
from jsonschema import validate, ValidationError
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import data_model
import set_train

# Set variable
def init():
    
    # Import json
    with open(Path(__file__).parent / "config_train.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Import Json schema
    with open(Path(__file__).parent / "config_train.schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    # validate
    try:
        validate(instance=cfg, schema=schema)
    except ValidationError as e:
        raise SystemExit(f"[CONFIG ERROR] {e.message}")
    
    # Set seed
    data_model.set_seed(cfg["seed"])
    
    # Set device
    device = data_model.pick_device(cfg["device"])
    
    return cfg, device

# Set Run
def set_run(cfg: dict, device):

    # Make output directory
    result_dir = set_train.init_set(cfg["out_dir"])
    
    # Dataset
    train_loader, val_loader, classes = data_model.prepare_dataset(cfg["train_dir"], cfg["val_dir"], cfg["batch_size"], cfg["img_size"], result_dir, device = device)
    
    # Build classification model
    model = data_model.build_model(len(classes), cfg["freeze_backbone"], cfg["pretrained"], dropout=cfg["dropout_p"], model_name=cfg.get("model_name", "resnet18"))
    model = model.to(device)

    return result_dir, train_loader, val_loader, classes, model

# Update model parameters
def set_optimizer(cfg: dict, model):
    
    # Loss function
    loss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"]
    )

    return loss, optimizer

# Resume weight loader
def set_resume(cfg: dict, model, optimizer, device):
    
    # Set 
    start_epoch = 0
    best_metric = 0.0
    
    resume_from = cfg.get("resume_from", None)
    weights_path = cfg.get("weights_path", None)
    
    # Checkpoint path
    if resume_from: 
        path = Path(resume_from)
        weights = torch.load(path, map_location="cpu")
        model.load_state_dict(weights)

    # Weights only
    elif weights_path:
        path = Path(weights_path)
        weights = torch.load(path, map_location="cpu")
        model.load_state_dict(weights)

    model = model.to(device)
    
    return model, optimizer, start_epoch, best_metric

# Train
def train(cfg: dict, model, train_loader, val_loader, loss, optimizer, device, result_dir: Path, start_epoch: int, best_metric: float):
    
    # setup
    epochs = cfg["epochs"]
    early_stop = cfg.get("early_stopping_enabled", False)    
    patience = cfg.get("early_stopping_patience", 10)
    target_accuracy = cfg.get("target_accuracy", None)
    
    checkpoint_every = cfg.get("checkpoint_every", None)
    
    # TensorBoard
    tb_enabled = cfg.get("tensorboard_enabled", False)
    if tb_enabled:
        tb_dir = Path(cfg.get("log_dir") or (result_dir / "tensorboard"))
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        writer = None
        
    history = {}
    bad_epochs = 0
    
    # Epoch loop
    for epoch in range(start_epoch, epochs):
        # Measure epoch execution time 
        t0 = time.time()

        epoch_human = epoch + 1
        print(f"epoch number:{epoch_human}")
        
        
        # Train
        model.train()
        train_loss_sum = 0.0
        
        # Batch train
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # reset gradient
            optimizer.zero_grad()
            
            # Loss, scalar number (tensor)
            outputs = model(images)
            batch_loss = loss(outputs, labels)
            
            # Update gradients and parameters ON
            batch_loss.backward()
            optimizer.step()
            
            train_loss_sum += batch_loss.item()
            
        train_loss_epoch = train_loss_sum / len(train_loader)

            
        # Validation
        model.eval()

        val_loss_sum = 0.0
        correct_predictions = 0
        total_prediction = 0
        
        # Batch val
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Loss, scalar number (tensor)
                outputs = model(images)
                val_loss = loss(outputs, labels)
                val_loss_sum += val_loss.item()
                
                # Prediction
                preds = torch.argmax(outputs, dim=1)
                correct_predictions  += (preds == labels).sum().item()
                total_prediction += labels.size(0)
            
        val_loss_epoch = val_loss_sum / len(val_loader)
        val_acc_epoch = correct_predictions / total_prediction
        
        epoch_time = time.time() - t0

        print(f"train_loss: {train_loss_epoch}\tval_loss: {val_loss_epoch}\tval_acc: {val_acc_epoch}\n")            
        
        # History
        set_train.update_history(history, train_loss_epoch, val_loss_epoch, val_acc_epoch, epoch_time)
        
        # Save Last model
        set_train.save_last_model(model, result_dir)
        
        # Save Best model
        if val_acc_epoch > best_metric:
            best_metric = val_acc_epoch
    
            set_train.save_best_model(model, result_dir)
            bad_epochs = 0
            
        else:
            bad_epochs += 1
        
        # Save snapshot weights
        if checkpoint_every and (epoch_human % checkpoint_every == 0):
            set_train.save_checkpoint(model, optimizer, result_dir, epoch_human, best_metric)
        
        # TensorBoard: every epoch 
        if writer is not None:
            set_train.tensorboard(writer, epoch_human, train_loss_epoch, val_loss_epoch, val_acc_epoch, epoch_time)


        # Early Stop
        if set_train.early_stop(val_acc_epoch, best_metric, bad_epochs, patience, early_stop, target_accuracy):
            print(f"\nearly stop {early_stop}")
            break
            

    if writer is not None:
        writer.close()
        
    set_train.save_history(history, result_dir)
    return history, best_metric
            
    
        
        
            
            
        
    
    



