# pt_train.py
# Commented out IPython magic to ensure Python compatibility.
from datetime import datetime
import pandas as pd
# from evaluation_plots import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassMatthewsCorrCoef
from sklearn.metrics import classification_report, balanced_accuracy_score
from data_loader import load_data_objs, CreateDataset_
from typing import Optional, Dict, Any, Tuple
from roc_curve_plot import plot_roc_curve
from precision_recall_plot import plot_precision_recall_curve
from confusion_matrix_plot import plot_confusion_matrix
from mlflow_logging import create_experiment
from mlflow.models import infer_signature
from torch.utils.data import DataLoader
from pt_engine import CustomTrainer
# from google.colab import userdata
from rich.progress import track
from pathlib import Path
import torch.multiprocessing as mp
import torch.distributed as dist
import plotly.io as pio
import torch.nn as nn
import numpy as np
import argparse
import random
import pickle
import torch
import time
import os

torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype])

from evaluation_plots import nn_model_plots
from model_builder import MulticlassClassifier_

pio.renderers.default = "colab"
pio.templates.default = "seaborn"


NUM_WORKERS = os.cpu_count()

with open('assets/artifacts/le_y/le_y.sav', 'rb') as f:
    le_y = pickle.load(f)



def find_free_port():
    """Finds a free port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to port 0 to get a free port
        print("Got free port...")
        return s.getsockname()[1]


def ddp_setup(rank: int, world_size: int) -> None:
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print("Init. process group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def model_eval(model: nn.Module, criterion: nn.CrossEntropyLoss, xval_path: str, yval_path: str, state_dict: Dict) -> Tuple[np.typing.NDArray, Any | float, torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
    model.load_state_dict(state_dict)
    Xval = np.load(xval_path, allow_pickle=True)
    yval = np.load(yval_path, allow_pickle=True)
    val_dts = CreateDataset_(Xval, yval)
    val_dtl = DataLoader(val_dts, batch_size=1,
                         shuffle=False, pin_memory=True,)
    classes_num = len(np.unique(yval))
    loss_fn = criterion.to(0)
    model.eval()
    acc_metric_func = MulticlassAccuracy(
        num_classes=classes_num, average="micro", sync_on_compute=False).to(0)
    f1_metric_func = MulticlassF1Score(
        num_classes=classes_num, average="macro", sync_on_compute=False).to(0)
    precision_metric_func = MulticlassPrecision(
        num_classes=classes_num, average="macro", sync_on_compute=False).to(0)
    recall_metric_func = MulticlassRecall(
        num_classes=classes_num, average="macro", sync_on_compute=False).to(0)
    mcc_metric_func = MulticlassMatthewsCorrCoef(
        num_classes=classes_num, sync_on_compute=False).to(0)
    pred_labels = np.array([{'targets': [], 'preds': []}])
    total_loss = 0
    total_samples_ = 0
    with torch.inference_mode():
        for source, targets in track(val_dtl, description="Evaluating...", style='red', complete_style='cyan', finished_style='green'):
            source = source.to(0)
            targets = targets.to(0)

            y_logits = model(source)
            preds = torch.softmax(y_logits, dim=1)
            preds = torch.argmax(preds, dim=1)
            loss = loss_fn(y_logits, targets)

            batch_size_ = source.size(0)  # Get batch size
            total_samples_ += batch_size_  # Accumulate total samples

            acc_metric_func.update(preds, targets)
            f1_metric_func.update(preds, targets)
            precision_metric_func.update(preds, targets)
            recall_metric_func.update(preds, targets)
            mcc_metric_func.update(preds, targets)
            total_loss += loss.item() * batch_size_

            pred_labels[0]['preds'].extend(
                preds.detach().cpu().numpy().tolist())
            pred_labels[0]['targets'].extend(targets.cpu().numpy().tolist())

        avg_loss = total_loss / total_samples_
        accuracy = acc_metric_func.compute()
        f1_score = f1_metric_func.compute()
        precision = precision_metric_func.compute()
        recall = recall_metric_func.compute()
        mcc = mcc_metric_func.compute()
        bal_acc = balanced_accuracy_score(
            pred_labels[0]['targets'], pred_labels[0]['preds'])


    print(f"\nLoss --> {avg_loss*100:.4f}% | Accuracy --> {accuracy*100:.4f}% | Balanced Acc --> {bal_acc*100:.4f}% | F1 Score --> {f1_score*100:.4f}% | Precision --> {precision*100:.4f}% | Recall --> {recall*100:.4f}% | Matthews Correlation Coefficient --> {mcc*100:.4f}%")
    print(
        f"\nClassification Report:\n{classification_report(pred_labels[0]['targets'], pred_labels[0]['preds'], target_names=le_y.classes_)}")

    return pred_labels, avg_loss, accuracy, f1_score, bal_acc, precision, recall, mcc


def main(rank: Optional[int], world_size: Optional[int], total_epochs: int, patience: int, batch_size: int, save_path: str | Path, xtrain_path: str, ytrain_path: str, xval_path: str, yval_path: str, learning_rate: float, lr_scheduler: str, use_gpu: bool) -> None:
    if use_gpu:
        # if rank == 0:
        print(f"{'>' * 10}MulticlassClassifier Model Training (GPU){'<' * 10}\n")
        ddp_setup(rank, world_size)  # type: ignore
    else:
        print(f"{'>' * 10}MulticlassClassifier Model Training (CPU){'<' * 10}\n")

    print("Initializing dataset and model...")
    train_dtl, val_dtl, model, criterion, optimizer = load_data_objs(batch_size, rank, world_size, xtrain_path, # type: ignore
                                                                    ytrain_path, xval_path, yval_path, use_gpu, learning_rate, NUM_WORKERS) # type: ignore
    print("Creating trainer...")
    trainer = CustomTrainer(model=model, train_data=train_dtl, val_data=val_dtl, criterion=criterion, optimizer=optimizer, gpu_id=rank, # type: ignore
                            save_path=save_path, use_gpu=use_gpu, patience=patience, max_epochs=total_epochs, world_size=world_size, lr_scheduler=lr_scheduler) # type: ignore
    print("Starting model training...")
    trainer.train()
    if use_gpu:
        cleanup()
    print(
        f"\n<{'=' * 10}Training completed & best model saved{'=' * 10}>\nExiting...")


def create_model_path(path_str: str) -> Optional[Path]:
    try:
        model_path = Path(path_str)
        model_path.mkdir(parents=True, exist_ok=True)

        # Check if the directory is writable
        if not os.access(model_path, os.W_OK):
            raise PermissionError(
                f"The directory {model_path} is not writable.")

        return model_path

    except (PermissionError, OSError) as e:
        print(f"Error creating model path: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def exec_time(st: float, et: float) -> None:
    hour = int(et-st)//3600
    minute = int((et-st) % 3600)//60
    second = int(et-st) % 60
    print(f'\nexec time => {hour:02d}hr : {minute:02d}min : {second:02d}sec')


if __name__ == "__main__":
    os.environ['NOTEBOOKAPP_IOPUB_MSG_RATE_LIMIT'] = '10000.0'
    os.environ['NOTEBOOKAPP_RATE_LIMIT_WINDOW'] = '10.0'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(
        description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int,
                        help='Total epochs to train the model (default: 10)')
    parser.add_argument('--patience', default=5, type=int,
                        help='Patience for increasing val_loss (default: 5)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Input batch size on each device (default: 32)')
    # parser.add_argument('--in_features', default=2, type=int,
                        # help='Number of classes (default: 2)')
    parser.add_argument('--model_save_path', default='./checkpoints', type=str,
                        help='Path to save the best model (default: ./checkpoints)')
    parser.add_argument('--xtrain_path', default='assets/x_train.npy', type=str,
                        help='Path to X_train numpy file (default: assets/x_train.npy)')
    parser.add_argument('--ytrain_path', default='assets/y_train.npy', type=str,
                        help='Path to y_train numpy file (default: assets/y_train.npy)')
    parser.add_argument('--xval_path', default='assets/x_val.npy', type=str,
                        help='Path to X_val numpy file (default: assets/x_val.npy)')
    parser.add_argument('--yval_path', default='assets/y_val.npy', type=str,
                        help='Path to y_val numpy file (default: assets/y_val.npy)')
    parser.add_argument('--xtest_path', default='assets/x_test.npy', type=str,
                        help='Path to X_test numpy file (default: assets/x_test.npy)')
    parser.add_argument('--ytest_path', default='assets/y_test.npy', type=str,
                        help='Path to y_test numpy file (default: assets/y_test.npy)')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Select learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler', default=None, type=str,
                        help='Select learning rate scheduler (default: None)')
    parser.add_argument('--world_size', default=None, type=int,
                        help='Pass the number of GPUs to be used for training (default: None(all))')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Train on GPU (default)')
    parser.add_argument('--no-gpu', dest='gpu',
                        action='store_false', help='Train on CPU')
    parser.set_defaults(gpu=True)
    args = parser.parse_args()

    MODEL_PATH = create_model_path(args.model_save_path)

    if MODEL_PATH is None:
        print("Failed to create model path. Exiting program.")
        exit(1)

    if args.use_gpu:
        if args.world_size is None:
            world_size = torch.cuda.device_count()
        elif args.world_size > 1 and torch.cuda.device_count() < args.world_size:
            print(
                f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} are available.")
            exit(1)
        else:
            world_size = args.world_size

        # Set the start method to 'forkserver'
        mp.set_start_method('forkserver', force=True)

        set_seed(42)

        start_time = time.time()
        mp.spawn(main, # type: ignore
                args=(world_size, args.total_epochs, args.patience, args.batch_size, MODEL_PATH, args.xtrain_path, args.ytrain_path,
                    args.xval_path, args.yval_path, args.learning_rate, args.lr_scheduler, args.use_gpu), nprocs=world_size, join=True)
        end_time = time.time()
        exec_time(start_time, end_time)
    else:
        start_time = time.time()
        set_seed(42)
        main(None, None, args.total_epochs, args.patience, args.batch_size, MODEL_PATH, args.xtrain_path, args.ytrain_path,
            args.xval_path, args.yval_path, args.learning_rate, args.lr_scheduler, args.use_gpu)
        end_time = time.time()
        exec_time(start_time, end_time)
