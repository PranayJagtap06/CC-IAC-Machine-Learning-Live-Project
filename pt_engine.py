# pt_engine.py
# Commented out IPython magic to ensure Python compatibility.
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, OneCycleLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import DataLoader
from rich.progress import track
from pathlib import Path
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def loss_metric_tensor(array: List[Dict[str, np.ndarray]]) -> torch.Tensor:
    all_tensors = [torch.tensor([[array[0][j][k] for k in range(
        len(array[0][j]))]], dtype=torch.float32) for j in array[0].keys()]
    b = torch.cat(all_tensors, dim=0)
    return b.transpose(0, 1)


class CustomTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_path: str | Path,
        use_gpu: bool,
        patience: int = 5,
        max_epochs: int = 10,
        world_size: int = 1,
        lr_scheduler: Optional[str] = None
        ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.classes_num = len(train_data.dataset.y.unique())  # type: ignore
        self.criterion = criterion
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_path = save_path
        self.use_gpu = use_gpu
        self.patience = patience
        self.max_epochs = max_epochs
        self.world_size = world_size
        self.lr_scheduler = lr_scheduler
        self.scheduler = self._create_scheduler()

        if self.use_gpu:
            if self.gpu_id is None:
                raise ValueError("gpu_id must be specified when using GPU.")
            self.model = DDP(self.model.to(self.gpu_id), device_ids=[self.gpu_id])
            if isinstance(self.train_data.sampler, torch.utils.data.DistributedSampler):
                self.train_sampler = self.train_data.sampler
            if isinstance(self.val_data.sampler, torch.utils.data.DistributedSampler):
                self.val_sampler = self.val_data.sampler

        self._setup_metrics()
        self._init_history()


    def _create_scheduler(self) -> Optional[_LRScheduler | ReduceLROnPlateau | OneCycleLR]:
        """Create a learning rate scheduler based on the configuration."""
        if self.lr_scheduler:
            if self.lr_scheduler == "reduce_lr":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=2
                )
            elif self.lr_scheduler == "one_cycle_lr":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=0.01,
                    epochs=self.max_epochs,
                    steps_per_epoch=len(self.train_data),
                    anneal_strategy="cos",
                )
            else:
                raise ValueError(
                    f"Invalid lr_scheduler value: {self.lr_scheduler}. "
                    "Valid options are: reduce_lr, one_cycle_lr"
                )
            return scheduler
        return None

    def _step_scheduler(self, epoch: int, val_loss: float) -> None:
        """Step the learning rate scheduler with appropriate arguments."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def _setup_metrics(self) -> None:
        if self.use_gpu:
            self.train_metric_accuracy = MulticlassAccuracy(num_classes=self.classes_num, average="micro", sync_on_compute=False).to(self.gpu_id)
            self.train_metric_f1score = MulticlassF1Score(num_classes=self.classes_num, average="macro", sync_on_compute=False).to(self.gpu_id)
            self.val_metric_accuracy = MulticlassAccuracy(num_classes=self.classes_num, average="micro", sync_on_compute=False).to(self.gpu_id)
            self.val_metric_f1score = MulticlassF1Score(num_classes=self.classes_num, average="macro", sync_on_compute=False).to(self.gpu_id)
        else:
            self.train_metric_accuracy = MulticlassAccuracy(num_classes=self.classes_num, average="micro")
            self.train_metric_f1score = MulticlassF1Score(num_classes=self.classes_num, average="macro")
            self.val_metric_accuracy = MulticlassAccuracy(num_classes=self.classes_num, average="micro")
            self.val_metric_f1score = MulticlassF1Score(num_classes=self.classes_num, average="macro")


    def _init_history(self) -> None:
        if self.use_gpu:
            self.train_losses_ = [{f'train_losses{i}': np.array([]) for i in range(self.world_size)}]
            self.val_losses_ = [{f'val_losses{i}': np.array([]) for i in range(self.world_size)}]
            self.train_f1s_ = [{f'train_f1s{i}': np.array([]) for i in range(self.world_size)}]
            self.val_f1s_ = [{f'val_f1s{i}': np.array([]) for i in range(self.world_size)}]
            self.train_accuracies_ = [{f'train_accs{i}': np.array([]) for i in range(self.world_size)}]
            self.val_accuracies_ = [{f'val_accs{i}': np.array([]) for i in range(self.world_size)}]
        else:
            self.train_losses_ = [{"train_losses": np.array([])}]
            self.val_losses_ = [{"val_losses": np.array([])}]
            self.train_f1s_ = [{"train_f1s": np.array([])}]
            self.val_f1s_ = [{"val_f1s": np.array([])}]
            self.train_accuracies_ = [{"train_accs": np.array([])}]
            self.val_accuracies_ = [{"val_accs": np.array([])}]


    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor, pred_labels: np.ndarray) -> tuple[Any, np.ndarray]:
        source = source.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        self.model.train()
        self.optimizer.zero_grad()
        y_logits = self.model(source)
        preds = torch.softmax(y_logits, dim=1)
        preds = torch.argmax(preds, dim=1)
        loss = self.criterion(y_logits, targets)
        loss.backward()
        self.optimizer.step()

        pred_labels[0]['preds'].extend(preds.detach().cpu().numpy().tolist())
        pred_labels[0]['targets'].extend(targets.cpu().numpy().tolist())

        self.train_metric_accuracy.update(preds, targets)
        self.train_metric_f1score.update(preds, targets)

        return loss.item(), pred_labels


    def _run_eval(self, epoch: int) -> Tuple[float, float, float, np.ndarray]:
        self.model.eval()
        total_loss = 0
        total_samples_ = 0

        if self.use_gpu:
            self.val_sampler.set_epoch(epoch)
            self.val_metric_accuracy.reset()
            self.val_metric_f1score.reset()

        pred_labels = np.array([{'targets': [], 'preds': []}])
        with torch.inference_mode():
            for source, targets in track(self.val_data, description="Evaluating...", style='red', complete_style='cyan', finished_style='green'):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                y_logits = self.model(source)
                preds = torch.softmax(y_logits, dim=1)
                preds = torch.argmax(preds, dim=1)
                loss = self.criterion(y_logits, targets)

                batch_size_ = source.size(0)  # Get batch size
                total_samples_ += batch_size_  # Accumulate total samples

                self.val_metric_accuracy.update(preds, targets)
                self.val_metric_f1score.update(preds, targets)
                total_loss += loss.item() * batch_size_

                pred_labels[0]['preds'].extend(preds.detach().cpu().numpy().tolist())
                pred_labels[0]['targets'].extend(targets.cpu().numpy().tolist())

        self.model.train()
        avg_loss = total_loss / total_samples_

        accuracy = self.val_metric_accuracy.compute()
        f1score = self.val_metric_f1score.compute()

        return avg_loss, accuracy.item(), f1score.item(), pred_labels


    def _run_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float, float, np.ndarray]:
        num_batches = len(self.train_data)
        total_loss = 0
        total_samples_ = 0

        if self.use_gpu:
            self.train_sampler.set_epoch(epoch)
            self.train_metric_accuracy.reset()
            self.train_metric_f1score.reset()

        pred_labels = np.array([{'targets': [], 'preds': []}])
        for source, targets in track(self.train_data,
                         description=f"""{f"[GPU{self.gpu_id}] " if self.use_gpu else ""}Epoch {epoch + 1}/{total_epochs} | Training: {num_batches} batches...""", style='red', complete_style='cyan', finished_style='green'):

            batch_size_ = source.size(0)  # Get batch size
            total_samples_ += batch_size_  # Accumulate total samples

            loss, pred_labels = self._run_batch(source, targets, pred_labels)
            total_loss += loss * batch_size_

        avg_loss = total_loss / total_samples_

        accuracy = self.train_metric_accuracy.compute()
        f1score = self.train_metric_f1score.compute()

        return avg_loss, accuracy.item(), f1score.item(), pred_labels


    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, train_accuracy: float, val_accuracy: float, train_f1score: float, val_f1score: float, train_labels: np.ndarray, val_labels: np.ndarray) -> None:
        model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        ckp = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_f1score': train_f1score,
            'val_f1score': val_f1score,
            'train_labels': train_labels,
            'val_labels': val_labels,
            }
        ckp_path = f"{self.save_path}/best_model.pt"
        torch.save(ckp, ckp_path)

        print(f"\t\tNew best model saved at {ckp_path} {f'from GPU{self.gpu_id} |' if self.use_gpu else ''} epoch {epoch+1}.")


    def gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        gathered_t = [torch.zeros_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_t, t)
        return torch.cat(gathered_t, dim=0)


    def _prepare_metrics_for_saving(self, metrics_dict: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        combined_dict = {}
        for d in metrics_dict:
            for key, value in d.items():
                combined_dict[key] = value
        return combined_dict


    def train(self) -> None:
        if self.use_gpu:
            should_stop = torch.zeros(1).to(self.gpu_id)
            patience_count = torch.zeros(1, dtype=torch.int32).to(self.gpu_id)

            # Gather losses from all GPUs
            train_losses = [torch.zeros(1).to(self.gpu_id)
                            for _ in range(self.world_size)]
            val_losses = [torch.zeros(1).to(self.gpu_id)
                          for _ in range(self.world_size)]
            train_f1s = [torch.zeros(1).to(self.gpu_id)
                         for _ in range(self.world_size)]
            val_f1s = [torch.zeros(1).to(self.gpu_id)
                       for _ in range(self.world_size)]
            train_accuracies = [torch.zeros(1).to(self.gpu_id)
                                for _ in range(self.world_size)]
            val_accuracies = [torch.zeros(1).to(self.gpu_id)
                              for _ in range(self.world_size)]
            val_losses_t = torch.empty(0).to(self.gpu_id)
            val_metrics_t = torch.empty(0).to(self.gpu_id)
        else:
            should_stop = torch.zeros(1)
            patience_count = torch.zeros(1, dtype=torch.int32)
            train_losses = []
            val_losses = []
            train_f1s = []
            val_f1s = []
            train_accuracies = []
            val_accuracies = []
            val_losses_t = []
            val_metrics_t = []

        set_seed(42)
        best_val_loss = float('inf')
        best_val_f1 = 0.0

        for epoch in range(self.max_epochs):
            train_loss, train_accuracy, train_f1, train_labels = self._run_epoch(epoch, self.max_epochs)
            val_loss, val_accuracy, val_f1, val_labels = self._run_eval(epoch)

            print(f"""\t{f"[GPU{self.gpu_id}] | " if self.use_gpu else ""}Batches: {len(self.train_data)} per GPU | Val Steps: {len(self.val_data)} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_accuracy: {train_accuracy:.4f} | val_accuracy: {val_accuracy:.4f} | train_f1: {train_f1:.4f} | val_f1: {val_f1:.4f} | Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}""")

            self._step_scheduler(epoch, val_loss)

            # ic(pred_labels[0]['targets'][-6:], pred_labels[0]['preds'][-6:])

            # Save losses for all GPUs
            if self.use_gpu:
                try:
                    torch.distributed.all_gather(
                        train_losses, torch.tensor([train_loss]).to(self.gpu_id))
                    torch.distributed.all_gather(
                        val_losses, torch.tensor([val_loss]).to(self.gpu_id))
                    torch.distributed.all_gather(
                        train_f1s, torch.tensor([train_f1]).to(self.gpu_id))
                    torch.distributed.all_gather(
                        val_f1s, torch.tensor([val_f1]).to(self.gpu_id))
                    torch.distributed.all_gather(
                        train_accuracies, torch.tensor([train_accuracy]).to(self.gpu_id))
                    torch.distributed.all_gather(
                        val_accuracies, torch.tensor([val_accuracy]).to(self.gpu_id))
                except RuntimeError as e:
                    print(f"Error gathering losses: {e}")
                    break

                for i in range(self.world_size):
                    self.train_losses_[0][f"train_losses{i}"] = np.append(
                        self.train_losses_[0][f"train_losses{i}"], train_losses[i].item())
                    self.val_losses_[0][f"val_losses{i}"] = np.append(
                        self.val_losses_[0][f"val_losses{i}"], val_losses[i].item())
                    self.train_f1s_[0][f"train_f1s{i}"] = np.append(
                        self.train_f1s_[0][f"train_f1s{i}"], train_f1s[i].item())
                    self.val_f1s_[0][f"val_f1s{i}"] = np.append(
                        self.val_f1s_[0][f"val_f1s{i}"], val_f1s[i].item())
                    self.train_accuracies_[0][f"train_accs{i}"] = np.append(
                        self.train_accuracies_[0][f"train_accs{i}"], train_accuracies[i].item())
                    self.val_accuracies_[0][f"val_accs{i}"] = np.append(
                        self.val_accuracies_[0][f"val_accs{i}"], val_accuracies[i].item())

                val_losses_t = loss_metric_tensor(self.val_losses_)
                val_metrics_t = loss_metric_tensor(self.val_f1s_)

                val_losses_last_item = np.min(val_losses_t[-1:].squeeze().numpy())
                val_metrics_last_item = np.max(val_metrics_t[-1:].squeeze().numpy())
                bval_loss = np.min(val_losses_t.numpy())
                bval_metric = np.max(val_metrics_t.numpy())

                improved = torch.tensor([False], dtype=torch.bool).to(self.gpu_id)
            else:
                self.train_losses_[0]["train_losses"] = np.append(self.train_losses_[0]["val_losses"], train_losses)
                self.train_f1s_[0]["train_f1s"] = np.append(self.train_f1s_[0]["val_f1s"], train_f1s)
                self.train_accuracies_[0]["train_accs"] = np.append(self.train_accuracies_[0]["val_accs"], train_accuracies)
                self.val_losses_[0]["val_losses"] = np.append(self.val_losses_[0]["val_losses"], val_losses)
                self.val_f1s_[0]["val_f1s"] = np.append(self.val_f1s_[0]["val_f1s"], val_f1s)
                self.val_accuracies_[0]["val_accs"] = np.append(self.val_accuracies_[0]["val_accs"], val_accuracies)

                val_losses_last_item = self.val_losses_[0]["val_losses"][-1]
                val_metrics_last_item = self.val_f1s_[0]["val_f1s"][-1]
                bval_loss = np.min(self.val_losses_[0]["val_losses"]) if len(self.val_losses_[0]["val_losses"]) > 0 else float('inf')
                bval_metric = np.max(self.val_f1s_[0]["val_f1s"]) if len(self.val_f1s_[0]["val_f1s"]) > 0 else 0.0
                improved = torch.tensor([False], dtype=torch.bool)

            if self.use_gpu:
                if (len(torch.where(val_losses_t == val_losses_last_item)[1]) == 1) and (
                        len(torch.where(val_metrics_t == val_metrics_last_item)[1]) == 1):
                    val_losses_last_gpu = torch.where(
                        val_losses_t == val_losses_last_item)[1].item()
                    val_metrics_last_gpu = torch.where(
                        val_metrics_t == val_metrics_last_item)[1].item()
                    val_losses_last_gpu_row = torch.where(
                        val_losses_t == val_losses_last_item)[0].item()
                    # val_metrics_last_gpu_row = torch.where(
                        # val_metrics_t == val_metrics_last_item)[0].item()

                    val_losses_last_metric = val_metrics_t[val_losses_last_gpu_row, val_losses_last_gpu] # type: ignore
                    # val_metrics_last_loss = val_losses_t[val_metrics_last_gpu_row, val_metrics_last_gpu] # type: ignore

                    if (val_losses_last_item == bval_loss) and (val_metrics_last_item == bval_metric) and (
                            val_losses_last_gpu == val_metrics_last_gpu) and (self.gpu_id == val_losses_last_gpu):
                        print(f"""\t\t1/1:[GPU{self.gpu_id}] val_loss improved to {
                        val_losses_last_item:.4f} | val_f1score improved to {val_metrics_last_item:.4f}""")
                        self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                        improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)

                        time.sleep(2)
                    elif (val_losses_last_item == bval_loss) and (val_metrics_last_item == bval_metric) and (
                            val_losses_last_gpu != val_metrics_last_gpu) and (self.gpu_id == val_losses_last_gpu):
                        print(f"""\t\t1/2:[GPU{self.gpu_id}] val_loss improved to {
                        val_losses_last_item:.4f} | val_f1score: {val_losses_last_metric:.4f}""")
                        self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                        improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)

                        time.sleep(2)
                    elif (val_losses_last_item == bval_loss) and (self.gpu_id == val_losses_last_gpu):
                        print(f"""\t\t1/3:[GPU{self.gpu_id}] val_loss improved to {
                        val_losses_last_item:.4f} | val_f1score: {val_losses_last_metric:.4f}""")
                        self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                        improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)

                        time.sleep(2)
                    # elif (val_metrics_last_item == bval_metric) and (self.gpu_id == val_metrics_last_gpu):
                    #     print(f"""\t\t1/4[GPU{self.gpu_id}] val_loss: {
                    #     val_metrics_last_loss:.4f} | val_f1score improved to {val_metrics_last_item:.4f}""")
                    #     self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                    #     improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)

                    #     time.sleep(2)
                elif (len(torch.where(val_losses_t == val_losses_last_item)[1]) == 1) and (
                        len(torch.where(val_metrics_t == val_metrics_last_item)[1]) > 1):
                    val_losses_last_gpu = torch.where(
                        val_losses_t == val_losses_last_item)[1].item()
                    val_losses_last_gpu_row = torch.where(
                        val_losses_t == val_losses_last_item)[0].item()
                    val_losses_last_metric = val_metrics_t[val_losses_last_gpu_row, val_losses_last_gpu] # type: ignore

                    if (val_losses_last_item == bval_loss) and (self.gpu_id == val_losses_last_gpu):
                        print(f"""\t\t3:[GPU{self.gpu_id}] val_loss improved to {
                        val_losses_last_item:.4f} | val_f1score: {val_losses_last_metric:.4f}""")
                        self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                        improved = torch.tensor([True], dtype=torch.bool).to(self.gpu_id)

                        time.sleep(2)
                else:
                    pass
            else:
                if (val_losses_last_item < best_val_loss) and (val_metrics_last_item > best_val_f1):
                    print(f"""\t\t1:val_loss improved to {
                    val_losses_last_item:.4f} | val_f1score improved to {val_metrics_last_item:.4f}""")
                    self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)
                    best_val_loss = val_losses_last_item
                    best_val_f1 = val_metrics_last_item

                    improved = torch.tensor([True], dtype=torch.bool)

                    time.sleep(2)
                elif val_losses_last_item < best_val_loss:
                    print(f"""\t\t2:val_loss improved to {
                    val_losses_last_item:.4f} | val_f1score: {val_metrics_last_item:.4f}""")
                    self._save_checkpoint(epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)
                    best_val_loss = val_losses_last_item

                    improved = torch.tensor([True], dtype=torch.bool)

                    time.sleep(2)
                # elif val_metrics_last_item == bval_metric:
                #     print(f"""\t\t3:val_loss: {
                #     val_losses_last_item:.4f} | val_f1score improved to {val_metrics_last_item:.4f}""")
                #     self._save_checkpoint(train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, train_labels, val_labels)

                #     improved = torch.tensor([True], dtype=torch.bool)

                #     time.sleep(2)
                else:
                    pass

            if self.use_gpu:
                # Synchronize patience count across all GPU
                improved_state = self.gather_tensor(improved)

                # Update patience count
                if self.world_size == 1:
                    if improved_state:
                        patience_count.zero_()
                    else:
                        patience_count += 1
                else:
                    if (improved_state[0] and improved_state[1]) or (improved_state[0] or improved_state[1]):
                        patience_count.zero_()
                    else:
                        patience_count += 1

                # Synchronize patience count across all GPUs
                all_patience_counts = self.gather_tensor(patience_count)
                max_patience_count = torch.max(all_patience_counts).item()
                patience_count.fill_(max_patience_count)

                if max_patience_count >= self.patience:
                    print(
                        f"\n[GPU{self.gpu_id}] Patience exceeded. Early stopping...")
                    should_stop[0] = 1

                # Synchronize the should_stop tensor across all GPUs
                should_stop_list = [torch.zeros(1).to(
                    self.gpu_id) for _ in range(self.world_size)]
                torch.distributed.all_gather(should_stop_list, should_stop)

                # If any GPU wants to stop, all GPUs should stop
                if any(_stop.item() for _stop in should_stop_list):
                    break
            else:
                if improved:
                    patience_count.zero_()
                else:
                    patience_count += 1

                    if patience_count >= self.patience:
                        print("\nPatience exceeded. Early stopping...")
                        break

            time.sleep(2)

        if self.use_gpu:
            # Ensure all GPUs exit the training loop together
            dist.barrier()

            if self.gpu_id == 0:
                metrics_to_save = np.array({
                    'train_losses': self._prepare_metrics_for_saving(self.train_losses_),
                    'train_f1s': self._prepare_metrics_for_saving(self.train_f1s_),
                    'train_accuracies': self._prepare_metrics_for_saving(self.train_accuracies_),
                    'val_losses': self._prepare_metrics_for_saving(self.val_losses_),
                    'val_f1s': self._prepare_metrics_for_saving(self.val_f1s_),
                    'val_accuracies': self._prepare_metrics_for_saving(self.val_accuracies_)
                })

                # Save all metrics in a single file
                np.save('assets/training_metrics.npy', metrics_to_save, allow_pickle=True)
        else:
            metrics_to_save = np.array({
                'train_losses': self._prepare_metrics_for_saving(self.train_losses_),
                'train_f1s': self._prepare_metrics_for_saving(self.train_f1s_),
                'train_accuracies': self._prepare_metrics_for_saving(self.train_accuracies_),
                'val_losses': self._prepare_metrics_for_saving(self.val_losses_),
                'val_f1s': self._prepare_metrics_for_saving(self.val_f1s_),
                'val_accuracies': self._prepare_metrics_for_saving(self.val_accuracies_)
            })

            np.save('assets/training_metrics.npy', metrics_to_save, allow_pickle=True)

