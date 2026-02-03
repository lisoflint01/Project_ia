from pathlib import Path

import train

def main() -> None:
    cfg, device = train.init()

    result_dir, train_loader, val_loader, classes, model = train.set_run(cfg, device)
    loss, optimizer = train.set_optimizer(cfg, model)

    model, optimizer, start_epoch, best_metric = train.set_resume(cfg, model, optimizer, device)

    history, best_metric = train.train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        optimizer=optimizer,
        device=device,
        result_dir=result_dir,
        start_epoch=start_epoch,
        best_metric=best_metric,
    )

if __name__ == "__main__":
    main()