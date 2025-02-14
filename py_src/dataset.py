import numpy as np
import torch
import random
import logging
from torch.utils.data import Dataset, DataLoader, Sampler

import py_src.third_party.shared_dataset.shareddataset as shared_mem_dataset
from py_src import internal_names, util
logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")

class LabelProbabilitySampler(Sampler):
    def __init__(self, labels, label_probs, indices_by_labels, num_samples):
        self.labels = labels
        self.label_probs = label_probs
        self.num_samples = num_samples
        self.indices = np.arange(len(labels))
        self.indices_by_labels = indices_by_labels

    def __iter__(self):
        sampled_indices = []
        for i in range(self.num_samples):
            random_number = random.uniform(0, 1)
            selected_label = None
            accumulated_prob = 0
            for single_label, label_probs in enumerate(self.label_probs):
                accumulated_prob += label_probs
                if random_number < accumulated_prob:
                    selected_label = single_label
                    break
            sampled_indices.append(random.choice(self.indices_by_labels[selected_label].tolist()))
        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples


class DatasetWithFastLabelSelection():
    def __init__(self, dataset: Dataset):
        self.raw_dataset = dataset
        self.labels = np.array(dataset.targets)
        indices = np.arange(len(self.labels))
        labels_set = set(self.labels)
        self.indices_by_labels = {}
        for single_label in labels_set:
            self.indices_by_labels[single_label] = indices[self.labels == single_label]

    """ dataloader with sampled label prob, to achieve iid/noniid """
    def get_train_loader_by_label_prob(self, label_prob, batch_size, worker=None) -> DataLoader:
        label_sampler = LabelProbabilitySampler(self.labels, label_prob, self.indices_by_labels, batch_size)
        if worker is None:
            # no shuffle need here because sampler provides shuffle
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, sampler=label_sampler, persistent_workers=True)
        else:
            # no shuffle need here because sampler provides shuffle
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, sampler=label_sampler, persistent_workers=True, num_workers=worker)
        return train_loader

    """ default dataloader without iid/noniid control """
    def get_train_loader_default(self, batch_size, worker=None) -> DataLoader:
        if worker is None:
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, persistent_workers=True, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, persistent_workers=True, shuffle=True, num_workers=worker)
        return train_loader

class DatasetInSharedMem(Dataset):
    def __init__(self, dataset: Dataset, shared_mem_name, transform=None):
        self.dataset_in_shared_mem = shared_mem_dataset.SharedDataset(dataset, shared_mem_name)
        self.transform = transform

        logger.info(f"loading shared memory dataset, shared memory name: {shared_mem_name}")
        for (image, label) in self.dataset_in_shared_mem:
            pass
        logger.info(f"loading shared memory dataset, shared memory name: {shared_mem_name} -- done")

    def __len__(self):
        return len(self.dataset_in_shared_mem)

    def __getitem__(self, idx):
        if idx >= len(self.dataset_in_shared_mem):
            raise StopIteration()
        sample, label = self.dataset_in_shared_mem.__getitem__(idx)
        if self.transform:
            sample = self.transform(sample)
            print("transform applied")
        return sample, label
