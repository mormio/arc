import argparse
import os

import neptune
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from arc import REPO_ROOT
from arc.arcdsl import PRIMITIVES
from arc.data import ARCDataLoader, REARCDataset, split_dataset
from arc.run.resnet import ARCResNetClassifier


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    run,
):
    best_f1 = 0.84
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            running_loss = 0.0
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            all_preds = []
            all_labels = []

            for batch in tqdm(loader, desc=phase):
                inputs = batch["combined_input"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(loader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average="samples")
            epoch_accuracy = accuracy_score(all_labels, all_preds)

            # Calculate TPR and TNR
            all_labels_flat = np.array(all_labels).flatten()
            all_preds_flat = np.array(all_preds).flatten()
            epoch_precision = precision_score(
                all_labels_flat, all_preds_flat, average="binary"
            )
            epoch_recall = recall_score(
                all_labels_flat, all_preds_flat, average="binary"
            )

            print(
                f"{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} Accuracy: {epoch_accuracy:.4f} "
                f"TPR: {epoch_precision:.4f} TNR: {epoch_recall:.4f}"
            )

            # log
            if run is not None:
                run[f"{phase}/epoch/loss"].append(epoch_loss)
                run[f"{phase}/epoch/f1"].append(epoch_f1)
                run[f"{phase}/epoch/accuracy"].append(epoch_accuracy)
                run[f"{phase}/epoch/precision"].append(epoch_precision)
                run[f"{phase}/epoch/recall"].append(epoch_recall)

            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        os.path.dirname(REPO_ROOT),
                        "models",
                        "resnet_rearc_bcelogits.pth",
                    ),
                )

    print(f"Best val F1: {best_f1:4f}")
    return model


def make_dataloaders(args):
    task_dir = os.path.join(REPO_ROOT, "data", "re_arc", "tasks")
    full_dataset = REARCDataset(task_dir=task_dir, debug=args.debug)

    if args.debug:
        full_dataset.data = full_dataset.data[:200]
    train_dataset, val_dataset = split_dataset(
        full_dataset, val_split=args.val_split, seed=0
    )
    train_loader = ARCDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        normalize=True,
        num_workers=min(4, os.cpu_count()),
    )
    val_loader = ARCDataLoader(
        val_dataset,
        normalize=True,
        num_workers=min(4, os.cpu_count()),
    )
    return train_loader, val_loader


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--description", type=str, default="", help="For neptune"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", help="architecture"
    )
    parser.add_argument("--loss", type=str, default="weightedBCE")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)

    # train and val datasets, loaders
    train_loader, val_loader = make_dataloaders(args)

    if not args.debug:
        run = neptune.init_run(
            project="mormio/arc",
            name="resnet_rearc",
            description=args.description,
        )
        run["parameters"] = {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "val_split": args.val_split,
            "seed": args.seed,
            "model": args.model,
            "loss": args.loss,
        }

    # init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ARCResNetClassifier(
        num_classes=len(PRIMITIVES),
    ).to(device)

    # train
    # criterion = nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(len(PRIMITIVES)) * 7
    ).to(
        device
    )  # roughly 7 negatives per positive in the dataset

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        run=None if args.debug else run,
    )
    if not args.debug:
        run.stop()


if __name__ == "__main__":
    main()
