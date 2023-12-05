import pickle

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Any, Dict, Union

from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID


def load_data() -> Tuple[Any, Any, Any]:
    """Load data from pickle files"""
    with open('data/train_data_rows.pkl', 'rb') as f:
        train_data_rows = pickle.load(f)

    with open('data/val_data_rows.pkl', 'rb') as f:
        val_data_rows = pickle.load(f)

    with open('data/test_data_rows.pkl', 'rb') as f:
        test_data_rows = pickle.load(f)

    print(f'Amount of train data: {len(train_data_rows)}')
    print(f'Amount of val data: {len(val_data_rows)}')
    print(f'Amount of test data: {len(test_data_rows)}')

    return train_data_rows, val_data_rows, test_data_rows


def id_to_one_hot_tensor(
        id_value: Union[int, torch.Tensor],
        num_classes: int
):
    """
    Convert an ID to a one-hot encoded tensor using PyTorch.

    Parameters:
    - id_value (int or Tensor): The ID value(s) to be converted to a one-hot tensor.
    - num_classes (int): Total number of classes/categories.

    Returns:
    - one_hot_tensor (Tensor): The one-hot encoded tensor.
    """
    # Convert int to tensor if single value
    if isinstance(id_value, int):
        id_value = torch.tensor(id_value)

    one_hot_tensor = torch.nn.functional.one_hot(id_value, num_classes=num_classes)
    return one_hot_tensor.to(torch.float32)


class CustomListDataset(Dataset):
    """Subclass of Dataset class"""

    def __init__(self, rows: List[Dict[str, Any]]):
        """ Initialize dataset.

        Args:
            rows: Data samples in a list of dict of features and labels.
        """
        self.data = rows
        self.n_traffic = len(PREFIX_TO_TRAFFIC_ID)
        self.n_app = len(PREFIX_TO_APP_ID)
        self.n_aux = len(AUX_ID)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        # Convert class index to one-hot encoding
        y_traffic = id_to_one_hot_tensor(d['traffic_label'], self.n_traffic)
        y_app = id_to_one_hot_tensor(d['app_label'], self.n_app)
        y_aux = id_to_one_hot_tensor(d['aux_label'], self.n_aux)

        # Concat a data sample including a sparse matrix converted
        sample = (torch.from_numpy(d['feature'].toarray()), y_traffic, y_app, y_aux)

        return sample


def get_dataset(data_rows: List[Dict[str, Any]]) -> Dataset:
    """Create a dataset with data samples"""
    ds = CustomListDataset(data_rows)
    return ds
