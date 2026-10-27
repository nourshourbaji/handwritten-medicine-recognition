import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.dataset import create_dataloaders
from src.models import build_model, count_parameters, init_weights


# Every experiment starts from these settings and changes only a few of them
DEFAULT_CONFIG = {
    "model": "cnn",
    "model_kwargs": {},
    "init": None,
    "dropout": 0.0,
    "loss": "cross_entropy",
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "l1": 0.0,
    "scheduler": None,
    "epochs": 20,
    "batch_size": 32,
    "augment": True,
    "normalize": None,
    "patience": None,
    "seed": 42,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MSELoss(nn.Module):
    """Squared error between the softmax probabilities and the one-hot target:
    sum over classes of (target - output)^2, like in the CNN training slides."""

    def forward(self, outputs, labels):
        probs = torch.softmax(outputs, dim=1)
        targets = F.one_hot(labels, probs.shape[1]).float()
        return ((targets - probs) ** 2).sum(dim=1).mean()


def get_criterion(name):
    losses = {
        # nn.CrossEntropyLoss applies softmax itself, so the model outputs raw scores
        "cross_entropy": nn.CrossEntropyLoss(),
        # softens the one-hot targets: 0.9 for the true class, rest shared (noise injection)
        "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        "mse": MSELoss(),
    }
    return losses[name]


def get_optimizer(name, parameters, lr, weight_decay):
    if name == "sgd":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(name, optimizer, epochs):
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device, l1=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # L1 regularization: add l1 * sum(|w|) to the loss
        # (L2 regularization is done by the optimizer's weight_decay)
        if l1 > 0:
            penalty = l1 * sum(p.abs().sum() for p in model.parameters())
            (loss + penalty).backward()
        else:
            loss.backward()

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(model, loaders, config, class_to_idx, checkpoint_path, device):
    criterion = get_criterion(config["loss"])
    if hasattr(model, "parameter_groups"):
        parameters = model.parameter_groups(
            lr=config["lr"],
            backbone_lr_mult=config.get(
                "backbone_lr_mult", 0.1
            )
        )
    
    else:
        parameters = model.parameters()
    
    optimizer = get_optimizer(
        config["optimizer"],
        parameters,
        config["lr"],
        config["weight_decay"]
    )
    scheduler = get_scheduler(config["scheduler"], optimizer, config["epochs"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = -1.0, 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["Training"], criterion, optimizer, device, config["l1"])
        val_loss, val_acc = evaluate(model, loaders["Validation"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f}")

        # Keep the weights of the epoch with the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_acc": val_acc,
            }, checkpoint_path)

        # Early stopping: stop if validation accuracy has not improved for `patience` epochs
        if config["patience"] and epoch - best_epoch >= config["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    return history, best_val_acc, best_epoch


def run_experiment(name, config, saved_splits, data_dir, checkpoint_dir, results_dir, device=None):
    """here we train one model and save its best checkpoint (.pt) and results (.json) to Drive
    If the results file already exists the experiment is skipped,
    so a notebook can be re-run after Colab disconnects."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {**DEFAULT_CONFIG, **config}

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_dir) / f"{name}.pt"
    results_path = Path(results_dir) / f"{name}.json"

    if results_path.exists():
        print(f"Skipping {name}: already done")
        return json.load(open(results_path))

    set_seed(config["seed"])

    loaders, class_to_idx = create_dataloaders(
        saved_splits, data_dir,
        batch_size=config["batch_size"],
        augment=config["augment"],
        normalize=config["normalize"]
    )

    model = build_model(config["model"], len(class_to_idx),
                        dropout=config["dropout"], **config["model_kwargs"])

    if config["init"] is not None:

        if (
            config["model"] == "transfer"
            and config["model_kwargs"].get("pretrained", True)
        ):
            raise ValueError(
                "Cannot initialize a pretrained model. "
                "Set init=None to preserve ImageNet weights."
            )

        init_weights(model, config["init"])
    
    model = model.to(device)

    print("=" * 70)
    print(f"Running: {name} | parameters: {count_parameters(model):,}")
    print("=" * 70)

    start = time.time()
    history, best_val_acc, best_epoch = train_model(
        model, loaders, config, class_to_idx, checkpoint_path, device
    )

    result = {
        "name": name,
        "config": config,
        "num_params": count_parameters(model),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_time_min": (time.time() - start) / 60,
        "history": history,
    }

    json.dump(result, open(results_path, "w"), indent=2)
    print(f"Best val acc {best_val_acc:.3f} at epoch {best_epoch}")
    return result
