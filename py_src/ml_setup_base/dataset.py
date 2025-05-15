import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from py_src.dataset import DatasetWithCachedOutputInSharedMem, DatasetWithCachedOutputInMem, ImageDatasetWithCachedInputInSharedMem

class DatasetSetup:
    def __init__(self, name, training_data, testing_data, labels=None):
        self.training_data = training_data
        self.testing_data = testing_data
        self.dataset_name = name

        if labels is None:
            self.labels = self._get_dataset_labels(self.testing_data)
        else:
            self.labels = labels
        sample_data = self.testing_data[0][0]
        self.tensor_size = sample_data.shape

    def _get_dataset_labels(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        labels_set = set()
        for _, labels in dataloader:
            labels_set.update(labels.tolist())
        return labels_set

""" Load env override file """
imagenet1k_path = None
imagenet100_path = None
imagenet10_path = None
dataset_env_file_path = f"{os.path.dirname(os.path.abspath(__file__))}/dataset_env.py"
if os.path.exists(dataset_env_file_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("dataset_env", dataset_env_file_path)
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)

    if hasattr(env, "imagenet1k_path"):
        imagenet1k_path = env.imagenet1k_path
        print("override imagenet1k_path: ", imagenet1k_path)
    if hasattr(env, "imagenet100_path"):
        imagenet100_path = env.imagenet100_path
        print("override imagenet100_path: ", env.imagenet100_path)
    if hasattr(env, "imagenet10_path"):
        imagenet10_path = env.imagenet10_path
        print("override imagenet10_path: ", env.imagenet10_path)

""" Helper functions """
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Calculate mean and variance
    mean = 0.
    var = 0.
    nb_samples = 0.

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    var /= nb_samples
    std = var ** 0.5
    return mean, std


""" MNIST """
def dataset_mnist():
    dataset_path = '~/dataset/mnist'
    dataset_name = "mnist"
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose([transforms.RandomRotation(5, fill=(0,)), transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    return DatasetSetup(dataset_name, train_data, test_data, labels=set(range(10)))


def dataset_mnist_224():
    dataset_path = '~/dataset/mnist'
    dataset_name = "mnist_224"
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    transforms_mnist_train = transforms.Compose(
        [transforms.RandomRotation(5, fill=(0,)),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[mean], std=[std])])
    transforms_mnist_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224, 224)),
         transforms.Normalize(mean=[mean], std=[std])])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms_mnist_train)
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms_mnist_test)
    return DatasetSetup(dataset_name, train_data, test_data, labels=set(range(10)))


""" Random MNIST """
def dataset_random_mnist():
    dataset_path = '~/dataset/random_mnist'
    dataset_name = "random_mnist"
    mnist_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transforms.Compose([transforms.ToTensor()]))
    mean, std = calculate_mean_std(mnist_train)
    mean, std = mean.mean().item(), std.mean().item()
    transforms_mnist_train = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    transforms_mnist_test = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    mnist_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transforms_mnist_train)
    mnist_test = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transforms_mnist_test)
    return DatasetSetup(dataset_name, mnist_train, mnist_test, labels=set(range(10)))


""" CIFAR10 """
def dataset_cifar10_32(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar10'
    dataset_name = "cifar10_32"
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    return DatasetSetup(dataset_name, cifar10_train, cifar10_test)

def dataset_cifar10_224(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar10'
    dataset_name = "cifar10_224"
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize(*stats)])
    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    return DatasetSetup(dataset_name, cifar10_train, cifar10_test)

def dataset_cifar100_32(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar100'
    dataset_name = "cifar100_32"
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    return DatasetSetup(dataset_name, cifar100_train, cifar100_test)

def dataset_cifar100_224(transforms_training=None, transforms_testing=None, mean_std = None):
    dataset_path = '~/dataset/cifar100'
    dataset_name = "cifar100_224"
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    if transforms_training is not None:
        transforms_cifar_train = transforms_training
    else:
        transforms_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(*stats)])
    if transforms_testing is not None:
        transforms_cifar_test = transforms_testing
    else:
        transforms_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize(*stats)])
    cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_cifar_train)
    cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_cifar_test)
    return DatasetSetup(dataset_name, cifar100_train, cifar100_test)

""" ImageNet """
def dataset_imagenet1k(transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet1k' if imagenet1k_path is None else imagenet1k_path
    dataset_name = "imagenet1k_224"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet1k_train", transform=final_transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet1k_test", transform=final_transforms_test)
    else:
        imagenet_train = datasets.ImageNet(root=dataset_path, split='train', transform=final_transforms_train)
        imagenet_test = datasets.ImageNet(root=dataset_path, split='val', transform=final_transforms_test)
    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 1000)))

def dataset_imagenet100(transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet100' if imagenet100_path is None else imagenet100_path
    dataset_name = "imagenet100_224"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet100_train", transform = final_transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet100_test", transform = final_transforms_test)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = final_transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = final_transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 100)))

def dataset_imagenet10(transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet10' if imagenet10_path is None else imagenet10_path
    dataset_name = "imagenet10_224"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if transforms_training is not None:
        final_transforms_train = transforms_training
    else:
        final_transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])
    if transforms_testing is not None:
        final_transforms_test = transforms_testing
    else:
        final_transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet10_train", transform = final_transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet10_test", transform = final_transforms_test)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = final_transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = final_transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 10)))