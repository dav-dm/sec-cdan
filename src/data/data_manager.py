import numpy as np
from sklearn.model_selection import train_test_split

from data.util import DataSplits
from data.data_reader import get_data_labels
from util.config import load_config


class DataManager:
    """
    Manages and prepares data for training, validation, and testing.
    """
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        
    def get_dataset_splits(self):
        # Loads source and target datasets based on provided arguments.
        dataset_args = {
            'num_pkts': self.args.num_pkts,
            'fields': self.args.fields,
            'is_flat': self.args.is_flat,
            'seed': self.args.seed,
        }
        
        # Splits the source dataset and, if present, also the target dataset.
        # It also sets the number of classes and checks for compatibility.
        src_dataset = get_data_labels(dataset=self.args.src_dataset, **dataset_args)
        src_splits, num_classes = self._train_val_test_split(dataset=src_dataset)
        
        if self.args.trg_dataset is not None:
            trg_dataset = get_data_labels(dataset=self.args.trg_dataset, **dataset_args)
            trg_splits, trg_num_classes = self._train_val_test_split(dataset=trg_dataset)
            
            assert num_classes == trg_num_classes, (
                'Mismatch between the classes of the source and target datasets'
            )            
        else:
            trg_splits = None
            
        return src_splits, trg_splits, num_classes
    
    
    def _train_val_test_split(self, dataset):
        """
        Splits the dataset into training, validation, and test sets.
        Conditionally includes quintuples if self.args.return_quintuple is True.
        """
        cf = load_config()
        return_quintuple = getattr(self.args, 'return_quintuple', False)
        
        arrays = [dataset['data'], dataset['labels']]
        if return_quintuple:
            arrays.append(dataset['quintuple'])

        # First split: entire dataset to train_val + test
        split1 = train_test_split(
            *arrays,
            train_size=cf['train_test_split'],
            random_state=self.seed,
            stratify=arrays[1]  # Always use full labels for stratification
        )
        train_val_arrays = split1[0::2]  # Every even-indexed element
        test_arrays = split1[1::2]       # Every odd-indexed element

        # Second split: train_val to train + val
        split2 = train_test_split(
            *train_val_arrays,
            train_size=cf['train_val_split'],
            random_state=self.seed,
            stratify=train_val_arrays[1]  # Stratify by train_val labels
        )
        train_arrays = split2[0::2]
        val_arrays = split2[1::2]

        if return_quintuple:
            return DataSplits(
                train=(train_arrays[0], train_arrays[1], train_arrays[2]), # data, labels, quintuple
                val=(val_arrays[0], val_arrays[1], val_arrays[2]),
                test=(test_arrays[0], test_arrays[1], test_arrays[2])
            ), len(np.unique(arrays[1]))
        else:
            return DataSplits(
                train=(train_arrays[0], train_arrays[1]), # data, labels
                val=(val_arrays[0], val_arrays[1]),
                test=(test_arrays[0], test_arrays[1])
            ), len(np.unique(arrays[1]))
