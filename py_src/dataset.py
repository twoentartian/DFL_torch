import numpy as np
import torch
import itertools
import random
import logging
from multiprocessing import shared_memory
from torch.utils.data import Dataset, DataLoader, Sampler

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

class SharedDataset(Dataset):
    """Dataset with lock-free shared memory cache (reused across processes).

    Wraps another Dataset, such that:
    - It gets samples from the wrapped Dataset the first time they're requested.
    - Afterwards, they are obtained from a shared memory cache.
    The memory cache uses multiprocessing.shared_memory.SharedMemory, so multiple
    processes will reuse it -- this is especially useful for very large datasets.

    To prevent unnecessary copies, the returned tensors point to the same shared
    memory. If you need to modify them in-place, copy them first, otherwise the
    buffer will be corrupted.

    Args:
      dataset (Dataset): A PyTorch Dataset to be wrapped. It needs to return either
        a Tensor or a tuple/list of Tensors.
      shared_mem_name (str): A name that identifies this shared memory buffer. Must
        be unique for each dataset, otherwise they will reuse the same memory and
        the buffer will be corrupted.

    Author: Joao F. Henriques
    """

    def __init__(self, dataset, shared_mem_name):
        self.dataset = dataset
        self.shared_mem_name = shared_mem_name
        self.shared_mem_size = None
        self._initialize()

    def _initialize(self):
        """Internal method to initialize the shared memory."""
        # get the first sample to figure out the returned layout and memory use
        sample = self.dataset[0]
        if torch.is_tensor(sample):  # wrap in a tuple
            sample = (sample,)
        try:  # ensure it's a tuple with tensors
            if not all(torch.is_tensor(e) for e in sample):
                raise TypeError()
        except TypeError:
            raise TypeError("SharedDataset: The wrapped dataset must return a Tensor or an iterable with Tensors.")

        self.sample_shapes = [e.shape for e in sample]
        self.sample_dtypes = [e.dtype for e in sample]
        sample_sizes = [len(e.numpy().tobytes()) for e in sample]

        # store offset where each tensor starts, in bytes.
        # e.g.: for 2 tensors of size 3, will contain [1, 4, 7] (the first byte is reserved)
        self.sample_offsets = list(itertools.accumulate([1] + sample_sizes))

        # total size of one sample
        total_sample_size = self.sample_offsets[-1]

        try:
            # load existing shared memory buffer if possible
            self.mem = shared_memory.SharedMemory(name=self.shared_mem_name, create=False)
            logger.info(f"loading existing shared memory {self.shared_mem_name} success")
        except FileNotFoundError:
            # doesn't exist, create it.
            shm_size = total_sample_size * len(self.dataset)
            self.shared_mem_size = shm_size
            shm_size = ((shm_size // 4096) + 1) * 4096
            logger.info(f"creating shared memory {self.shared_mem_name} with size {shm_size/1024**3}GB")
            self.mem = shared_memory.SharedMemory(name=self.shared_mem_name, create=True,
                                                  size=shm_size)

            # initialize those single-byte flags with 0.
            # it's OK if this overwrites the progress of some concurrent processes; it just means a bit more loading overhead.
            self.mem.buf[:self.shared_mem_size:total_sample_size] = bytes(len(self.dataset))


    def __getitem__(self, index):
        """Return a single sample. Samples will be cached in the SharedMemory buffer
        and reused if possible.

        Args:
          index (int): Index of item.

        Returns:
          Tensor: The image.
        """
        if index >= len(self.dataset):
            raise StopIteration()

        if self.mem is None:  # may happen after serializing the dataset
            self._initialize()

        start_byte = self.sample_offsets[-1] * index

        if self.mem.buf[start_byte] == 0:
            # sample not loaded yet, load it from the wrapped Dataset
            sample = self.dataset[index]
            if torch.is_tensor(sample):
                sample = (sample,)  # wrap in tuple

            # cache each of the sample's tensors in the SharedMemory
            for i in range(len(self.sample_shapes)):
                start = start_byte + self.sample_offsets[i]
                end = start_byte + self.sample_offsets[i + 1]
                if (end - start) != sample[i].numel() * sample[i].element_size():
                    raise ValueError(f"Size mismatch at index {i}")
                self.mem.buf[start:end] = sample[i].view(-1).numpy().tobytes()

            # finally, record that this sample has been loaded, in the first byte
            self.mem.buf[start_byte] = 1

        else:
            # the sample is cached, so extract it and convert to a tuple of tensors
            sample = []
            for i in range(len(self.sample_shapes)):
                start = start_byte + self.sample_offsets[i]
                end = start_byte + self.sample_offsets[i + 1]
                tensor = torch.frombuffer(self.mem.buf[start:end], dtype=self.sample_dtypes[i])
                sample.append(tensor.view(self.sample_shapes[i]))
            sample = tuple(sample)

        return sample

    def __len__(self):
        """Return the length of the dataset (number of samples). Defers to the wrapped Dataset.

        Returns:
          int: Number of samples.
        """
        return len(self.dataset)

    def __del__(self):
        """Close the SharedMemory handle on exit."""
        if hasattr(self, 'mem') and self.mem is not None:
            self.mem.close()

    def __getstate__(self):
        """Serialize without the SharedMemory references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state['mem'] = None
        return state

    @classmethod
    def unlink(C, shared_mem_name):
        """Class method to unlink the SharedMemory buffer with a given name.
        Can be used in case the OS fails to release the buffer when it is not
        being used by any process. Should not be needed in normal operation."""
        try:
            mem = shared_memory.SharedMemory(name=shared_mem_name, create=False)
            mem.unlink()
            mem.close()
        except FileNotFoundError:
            print('SharedDataset: SharedMemory was not initialized, so unlinking has no effect.')


class DatasetInSharedMem(Dataset):
    def __init__(self, dataset: Dataset, shared_mem_name, transform=None):
        self.dataset_in_shared_mem = SharedDataset(dataset, shared_mem_name)
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
