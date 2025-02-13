import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler


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

    def get_train_loader_by_label_prob(self, label_prob, batch_size, worker=None) -> DataLoader:
        label_sampler = LabelProbabilitySampler(self.labels, label_prob, self.indices_by_labels, batch_size)
        if worker is None:
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, sampler=label_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(self.raw_dataset, batch_size=batch_size, sampler=label_sampler, num_workers=worker)
        return train_loader

