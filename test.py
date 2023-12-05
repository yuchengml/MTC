import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtc import MTC
from mtt import MTT
from one_d_cnn import OneDCNN
from tcn import TCN
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score, \
    MulticlassConfusionMatrix
from tqdm import tqdm
from typing import List

from dataset import get_dataset, load_data
from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID


def test_op(
        model: nn.Module,
        batch_size: int = 128,
        device: str = 'cuda:0',
):
    """ Perform testing.

    Args:
        model: The model for testing.
        batch_size: Batch size.
        device: Device number to serve model.

    Returns:
        Metrics for tasks including the average and the per-class results.
    """
    # Check if GPU is available, otherwise switch to CPU
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'

    # Load test data and ignore training or validation data
    _, _, test_data_rows = load_data()
    test_dataset = get_dataset(test_data_rows)

    # DataLoader to load the data in batches
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move model to specified device (GPU/CPU)
    model = model.to(device)
    model.eval()

    # Initialize output lists for Tasks
    task1_outputs = []
    task2_outputs = []
    task3_outputs = []

    # Iterate through test data
    with torch.no_grad():
        pbar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc=f"Testing")
        for batch_idx, (inputs, labels_task1, labels_task2, labels_task3) in pbar:
            inputs = inputs.to('cuda:0')
            outputs1, outputs2, outputs3 = model(inputs)

            # Move outputs back to CPU
            outputs1 = outputs1.cpu()
            outputs2 = outputs2.cpu()
            outputs3 = outputs3.cpu()

            # Append task outputs
            task1_outputs.append((outputs1, labels_task1))
            task2_outputs.append((outputs2, labels_task2))
            task3_outputs.append((outputs3, labels_task3))

    # Initialize a list to store computed metrics for each task
    task_metrics = []
    for task_outputs, n_classes in zip([task1_outputs, task2_outputs, task3_outputs],
                                       [len(PREFIX_TO_TRAFFIC_ID), len(PREFIX_TO_APP_ID), len(AUX_ID)]):
        # Variable to compute total loss
        total_loss = 0.0
        total_batches = 0

        # Initialize evaluation metrics for precision, recall, F1 score, accuracy, and confusion matrix
        prec_metric = MulticlassPrecision(average=None, num_classes=n_classes)
        recall_metric = MulticlassRecall(average=None, num_classes=n_classes)
        f1_metric = MulticlassF1Score(average=None, num_classes=n_classes)
        accuracy_metric = MulticlassAccuracy(num_classes=n_classes)
        conf_mat_metric = MulticlassConfusionMatrix(num_classes=n_classes)

        # Calculate evaluation metrics for each output and label in the task
        for outputs, labels in task_outputs:
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

            # Update evaluation metrics
            prec_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            recall_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            f1_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            accuracy_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
            conf_mat_metric.update(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

        # Compute average loss and metrics for the task
        avg_loss = total_loss / total_batches
        avg_precision = prec_metric.compute().mean().detach().cpu().numpy()
        avg_recall = recall_metric.compute().mean().detach().cpu().numpy()
        avg_f1 = f1_metric.compute().mean().detach().cpu().numpy()
        accuracy = accuracy_metric.compute().detach().cpu().numpy()
        conf_mat = conf_mat_metric.normalized("pred").detach().cpu().numpy()

        # Compute per-class precision, recall, and F1 score
        per_cls_precision = prec_metric.compute().detach().cpu().numpy()
        per_cls_recall = recall_metric.compute().detach().cpu().numpy()
        per_cls_f1 = f1_metric.compute().detach().cpu().numpy()

        task_metrics.append((avg_loss, avg_precision, avg_recall, avg_f1, accuracy, conf_mat,
                             per_cls_precision, per_cls_recall, per_cls_f1))

    return task_metrics


def draw_heat_map(
        matrix: np.array,
        x_labels: List[str],
        y_labels: List[str],
        fig_name: str,
        output_file: str
):
    """ Draw and save the heatmap.

    Args:
        matrix: Multi-class confusion matrix in numpy array.
        x_labels: List of labels.
        y_labels: List of labels.
        fig_name: Figure name.
        output_file: Output file name.

    Returns:
        None.
    """
    plt.figure(figsize=(21, 9))
    heatmap = plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title(f'Confusion Matrix: {fig_name}')

    # Set custom x and y labels
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.yticks(np.arange(len(y_labels)), y_labels)

    # Save the plot as an image file
    plt.savefig(output_file, bbox_inches='tight')  # Save as 'heatmap.png' in the current directory

    # plt.show()


def testing():
    # Models for testing
    testing_models = [OneDCNN, MTT, TCN, MTC]
    model_metrics = dict()

    # Testing by each model to collect metrics
    for module in testing_models:
        m = module()
        print(f'Test `{m.__class__.__name__}` model...')
        m.load_state_dict((torch.load(f'{m.__class__.__name__}_model.pt')))
        metrics = test_op(m)
        model_metrics[m.__class__.__name__] = metrics
        # summary(m, input_size=(128, 1, 1500))

    # Convert metrics to formatted records
    task_1_records = []
    task_2_records = []
    task_3_records = []
    for model_name, metrics in model_metrics.items():
        # Task 1
        t1_metrics = metrics[0]
        r = {
            'model': model_name,
            'avg_loss': np.round(t1_metrics[0], 2),
            'avg_precision': np.round(t1_metrics[1], 2),
            'avg_recall': np.round(t1_metrics[2], 2),
            'avg_f1': np.round(t1_metrics[3], 2),
            'accuracy': np.round(t1_metrics[4], 2),
        }
        for k, v in zip(list(PREFIX_TO_TRAFFIC_ID.keys()), list(t1_metrics[6])):
            r[f'precision_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_TRAFFIC_ID.keys()), list(t1_metrics[7])):
            r[f'recall_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_TRAFFIC_ID.keys()), list(t1_metrics[8])):
            r[f'f1_{k}'] = np.round(v, 2)

        task_1_records.append(r)

        draw_heat_map(t1_metrics[5], PREFIX_TO_TRAFFIC_ID.keys(), PREFIX_TO_TRAFFIC_ID.keys(),
                      fig_name=f'{model_name} (Traffic)',
                      output_file=f'results/confusion_matrix_{model_name}_traffic.jpg')

        # Task 2
        t2_metrics = metrics[1]
        r = {
            'model': model_name,
            'avg_loss': np.round(t2_metrics[0], 2),
            'avg_precision': np.round(t2_metrics[1], 2),
            'avg_recall': np.round(t2_metrics[2], 2),
            'avg_f1': np.round(t2_metrics[3], 2),
            'accuracy': np.round(t2_metrics[4], 2)
        }
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t2_metrics[6])):
            r[f'precision_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t2_metrics[7])):
            r[f'recall_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t2_metrics[8])):
            r[f'f1_{k}'] = np.round(v, 2)

        task_2_records.append(r)

        draw_heat_map(t2_metrics[5], PREFIX_TO_APP_ID.keys(), PREFIX_TO_APP_ID.keys(),
                      fig_name=f'{model_name} (App.)',
                      output_file=f'results/confusion_matrix_{model_name}_app.jpg')

        # Task 3
        t3_metrics = metrics[1]
        r = {
            'model': model_name,
            'avg_loss': np.round(t3_metrics[0], 2),
            'avg_precision': np.round(t3_metrics[1], 2),
            'avg_recall': np.round(t3_metrics[2], 2),
            'avg_f1': np.round(t3_metrics[3], 2),
            'accuracy': np.round(t3_metrics[4], 2)
        }
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t3_metrics[6])):
            r[f'precision_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t3_metrics[7])):
            r[f'recall_{k}'] = np.round(v, 2)
        for k, v in zip(list(PREFIX_TO_APP_ID.keys()), list(t3_metrics[8])):
            r[f'f1_{k}'] = np.round(v, 2)

        task_3_records.append(r)

        draw_heat_map(t3_metrics[5], AUX_ID.keys(), AUX_ID.keys(),
                      fig_name=f'{model_name} (Aux.)',
                      output_file=f'results/confusion_matrix_{model_name}_aux.jpg')

    # Output records to CSV files
    pd.DataFrame.from_records(task_1_records).to_csv('results/classification_result_traffic.csv')
    pd.DataFrame.from_records(task_2_records).to_csv('results/classification_result_app.csv')
    pd.DataFrame.from_records(task_3_records).to_csv('results/classification_result_aux.csv')


if __name__ == '__main__':
    testing()
