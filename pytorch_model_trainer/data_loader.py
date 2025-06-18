# data_loader.py
# Commented out IPython magic to ensure Python compatibility.
# %%writefile data_loader.py
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from model_builder import MulticlassClassifier_
from torch.utils.data import DataLoader

# from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

# from sklearn.utils.class_weight import compute_class_weight


def get_num_workers() -> int:
    cpu_count = os.cpu_count()
    return cpu_count if isinstance(cpu_count, int) and cpu_count > 0 else 1


# Type annotation is now correct since get_num_workers always returns an int
NUM_WORKERS: int = get_num_workers()


class CreateDataset_(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X) == len(y), "X and y must have the same length"
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.x):
            raise IndexError(
                f"Index {idx} is out of bounds for dimension 0 with size {len(self.x)}"
            )
        return self.x[idx], self.y[idx]


def load_data_objs(
    batch_size: int,
    rank: int,
    world_size: int,
    # epochs: int,
    # in_features: int,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    use_gpu: bool,
    # gpu_id: int,
    learning_rate: float,
    num_workers: int = NUM_WORKERS,
    # lr_scheduler: Optional[str] = None,
) -> tuple[
    DataLoader, DataLoader, nn.Module, nn.CrossEntropyLoss, torch.optim.Optimizer
]:
    x_train = np.load(x_train_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)
    x_val = np.load(x_val_path, allow_pickle=True)
    y_val = np.load(y_val_path, allow_pickle=True)
    train_dts = CreateDataset_(x_train, y_train)
    val_dts = CreateDataset_(x_val, y_val)

    model = MulticlassClassifier_(
        in_features=x_train.shape[1], classes_num=len(train_dts.y.unique())
    )

    if use_gpu:
        dist_sampler_train = DistributedSampler(
            train_dts, num_replicas=world_size, rank=rank, seed=42
        )
        train_dtl = DataLoader(
            train_dts,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=dist_sampler_train,
            num_workers=num_workers,
        )

        dist_sampler_val = DistributedSampler(
            val_dts, num_replicas=world_size, rank=rank, seed=42
        )
        val_dtl = DataLoader(
            val_dts,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            sampler=dist_sampler_val,
            num_workers=num_workers,
        )

    else:
        train_dtl = DataLoader(
            train_dts,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        val_dtl = DataLoader(
            val_dts,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    # scheduler = None
    # if lr_scheduler:
    #     LR_SCHEDULER = {
    #         # requires to set metric
    #         "reduce_lr": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2),
    #         "one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_dtl), anneal_strategy='cos')
    #     }

    #     if lr_scheduler in LR_SCHEDULER:
    #         scheduler = LR_SCHEDULER[lr_scheduler]
    #     else:
    #         raise ValueError(f"""Invalid lr_scheduler value: {
    #             lr_scheduler}. Valid options are: {list(LR_SCHEDULER.keys())}""")

    return train_dtl, val_dtl, model, criterion, optimizer
