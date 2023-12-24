import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm
from typing import Tuple, List, Any

from dataset import get_dataset, load_data
from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID


def evaluate_op(
        model: nn.Module,
        data_loader: DataLoader,
        device: str = 'cuda:0'
) -> List[Tuple[float, Any, Any, Any, Any]]:
    """ Perform evaluation.

    Args:
        model: Model instance.
        data_loader: Data loader in PyTorch.
        device: Device name/number for usage. The desired device of the parameters
                and buffers in this module.

    Returns:
        Task metrics
    """
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'

    # Set the model to evaluation mode
    model.eval()

    # Initialize output lists for tasks
    task1_outputs = []
    task2_outputs = []
    task3_outputs = []

    # Disable gradient calculation as it's unnecessary during evaluation
    with torch.no_grad():
        # Iterate through each batch in the data_loader with a progress bar for evaluation
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluation")
        for batch_idx, (inputs, labels_task1, labels_task2, labels_task3) in pbar:
            # Move input data to the GPU (if available)
            inputs = inputs.to(device)

            # Get outputs for tasks from the model
            outputs1, outputs2, outputs3 = model(inputs)

            # Move output data back to CPU
            outputs1 = outputs1.cpu()
            outputs2 = outputs2.cpu()
            outputs3 = outputs3.cpu()

            # Append task outputs along with their corresponding labels to the lists
            task1_outputs.append((outputs1, labels_task1))
            task2_outputs.append((outputs2, labels_task2))
            task3_outputs.append((outputs3, labels_task3))

    # Initialize a list for task metrics
    task_metrics = []
    for task_outputs, n_classes in zip([task1_outputs, task2_outputs, task3_outputs],
                                       [len(PREFIX_TO_TRAFFIC_ID), len(PREFIX_TO_APP_ID), len(AUX_ID)]):
        total_loss = 0.0
        total_batches = 0

        # Initialize metrics for multiclass
        prec_metric = MulticlassPrecision(average=None, num_classes=n_classes)
        recall_metric = MulticlassRecall(average=None, num_classes=n_classes)
        f1_metric = MulticlassF1Score(average=None, num_classes=n_classes)
        accuracy_metric = MulticlassAccuracy(num_classes=n_classes)

        # Calculate metrics for each output and label in the task
        for outputs, labels in task_outputs:
            # Calculate cross-entropy loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            total_batches += 1
            # print((outputs.shape, labels.shape))

            # Update metrics
            prec_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            recall_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            f1_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            accuracy_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

        # Compute average loss and metrics
        avg_loss = total_loss / total_batches
        avg_precision = prec_metric.compute().mean()
        avg_recall = recall_metric.compute().mean()
        avg_f1 = f1_metric.compute().mean()
        accuracy = accuracy_metric.compute().detach().cpu().numpy()

        # Append computed metrics to the task metrics list
        task_metrics.append((avg_loss, avg_precision, avg_recall, avg_f1, accuracy))

    return task_metrics


def train_op(
        model: nn.Module,
        batch_size: int = 128,
        n_epochs: int = 100,
        device: str = 'cuda:0',
        task_weights: Tuple[int, int] = (1, 1),
        use_wandb: bool = False
):
    """ Perform training iteration in PyTorch.

    Args:
        model: Model instance.
        batch_size: Batch size.
        n_epochs: Epochs.
        device: Device name/number for usage. The desired device of the parameters
                and buffers in this module.
        task_weights: Assign weights of loss calculation for multi-class classification.
        use_wandb: Enable wandb to record training log.

    Returns:

    """
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'

    assert len(task_weights) == 3, 'Length of task weights should be 3'

    # Load raw data
    train_data_rows, val_data_rows, test_data_rows = load_data()

    # Prepare dataset from subclass of torch.utils.data.Dataset
    train_dataset = get_dataset(train_data_rows)
    val_dataset = get_dataset(val_data_rows)

    # Create DataLoader to load the data in batches
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Move model parameters to specified devices
    model = model.to(device)

    # Create optimizer with gradient decent method for training
    # The author of `MTC` used weight decay to avoid over-fitting
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Use wandb to record training log
    if use_wandb:
        import wandb
        run = wandb.init(project="MTC")
        config = run.config
        config['model'] = model.__class__.__name__
        config['dataset'] = "ISCX"
        run.watch(model)
        log_interval = 1000
    else:
        run = None
        log_interval = None

    # Initialize accuracy to save best model
    best_accuracy = 0.0
    avg_loss = 0.0

    # Training loop by epoch
    for epoch in range(n_epochs):
        # Store
        running_loss = 0.0

        # Set training mode
        model.train()

        # Initialize progress bar
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch + 1}, Loss: 0.000")

        # Loop training dataset
        for batch_idx, (inputs, labels_task1, labels_task2, labels_task3) in pbar:
            # Move model parameters to specified devices
            inputs = inputs.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            # Forward pass
            outputs_task1, outputs_task2, outputs_task3 = model(inputs)

            # Get loss values
            loss_task1 = F.cross_entropy(outputs_task1, labels_task1.to(device))
            loss_task2 = F.cross_entropy(outputs_task2, labels_task2.to(device))
            loss_task3 = F.cross_entropy(outputs_task3, labels_task3.to(device))

            # Calculate weighted loss
            batch_loss = (task_weights[0] * loss_task1 + task_weights[1] * loss_task2 + task_weights[
                2] * loss_task3) / sum(task_weights)

            # Backpropagation and update model parameters
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            # Update the description of progress bar with the average loss
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            pbar.set_postfix(loss=avg_loss)

            # Log loss to wandb if enable
            if use_wandb:
                if batch_idx % log_interval == 0:
                    run.log({"loss": avg_loss})

        # Evaluation after each epoch
        metrics = evaluate_op(model, val_data_loader)

        # Log metrics
        task_accuracy = []
        for task_i, m in enumerate(metrics):
            print(f"Task {task_i + 1} - Validation Loss: {m[0]:.4f}, "
                  f"Precision: {m[1]:.4f}, Recall: {m[2]:.4f}, F1: {m[3]:.4f} , Accuracy: {m[4]:.4f}")

            # Log task metrics to wandb if enable
            if use_wandb:
                run.log({f'task_{task_i}/loss': m[0]})
                run.log({f'task_{task_i}/precision': m[1]})
                run.log({f'task_{task_i}/recall': m[2]})
                run.log({f'task_{task_i}/f1': m[3]})
                run.log({f'task_{task_i}/accuracy': m[4]})

            # Record task accuracy
            task_accuracy.append(m[4])

        # Save best model according to the accuracy of 'application'
        target_accuracy = task_accuracy[1]
        if target_accuracy >= best_accuracy:
            torch.save(model.state_dict(), f'{model.__class__.__name__}_model.pt')
            # Update best accuracy
            best_accuracy = target_accuracy

        # Update scheduler to modify learning rate
        scheduler.step(avg_loss)
