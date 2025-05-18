import os
from enum import Enum, auto
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from py_src.dataset import DatasetWithCachedOutputInSharedMem, DatasetWithCachedOutputInMem, ImageDatasetWithCachedInputInSharedMem
from py_src.ml_setup_base.base import DatasetSetup
from torchvision.transforms.autoaugment import TrivialAugmentWide
from torchvision.transforms.v2 import RandAugment



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


""" Dataset Enum """
class DatasetType(Enum):
    default = auto()
    mnist = auto()
    mnist_224 = auto()
    random_mnist = auto()
    cifar10 = auto()
    cifar10_224 = auto()
    cifar100 = auto()
    cifar100_224 = auto()
    imagenet10 = auto()
    imagenet100 = auto()
    imagenet1k = auto()

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
def dataset_mnist(rescale_to_224=False, random_rotation=5):
    dataset_path = '~/dataset/mnist'
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    if rescale_to_224:
        dataset_name = str(DatasetType.mnist_224)
    else:
        dataset_name = str(DatasetType.mnist)

    train_transforms = []
    test_transforms = []
    # data augmentation
    if random_rotation != 0:
        train_transforms.append(transforms.RandomRotation(random_rotation, fill=(0,)))
    if rescale_to_224:
        train_transforms.append(transforms.Resize((224, 224)))
        test_transforms.append(transforms.Resize((224, 224)))
    else:
        train_transforms.append(transforms.RandomCrop(28, padding=2))
    train_transforms = train_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    test_transforms = test_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms.Compose(train_transforms))
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms.Compose(test_transforms))
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
def dataset_cifar10(rescale_to_224=False, transforms_training=None, transforms_testing=None, mean_std=None):
    dataset_path = '~/dataset/cifar10'
    if rescale_to_224:
        dataset_name = str(DatasetType.cifar10_224)
    else:
        dataset_name = str(DatasetType.cifar10)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    train_transforms = []
    test_transforms = []
    # data augmentation
    data_augmentation = [transforms.RandomHorizontalFlip(p=0.5)]
    if rescale_to_224:
        train_transforms.append(transforms.Resize((224, 224)))
        test_transforms.append(transforms.Resize((224, 224)))
    else:
        train_transforms.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
    train_transforms = train_transforms + data_augmentation
    train_transforms = train_transforms + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = test_transforms + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_training)
    if transforms_testing is None:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_testing)

    return DatasetSetup(dataset_name, cifar10_train, cifar10_test, labels=set(range(10)))

""" CIFAR100 """
def dataset_cifar100(rescale_to_224=False, transforms_training=None, transforms_testing=None, mean_std=None):
    dataset_path = '~/dataset/cifar100'
    if rescale_to_224:
        dataset_name = str(DatasetType.cifar100_224)
    else:
        dataset_name = str(DatasetType.cifar100)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    train_transforms = []
    test_transforms = []
    # data augmentation
    data_augmentation = [transforms.RandomHorizontalFlip(p=0.5)]
    if rescale_to_224:
        train_transforms.append(transforms.Resize((224, 224)))
        test_transforms.append(transforms.Resize((224, 224)))
    else:
        train_transforms.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
    train_transforms = train_transforms + data_augmentation
    train_transforms = train_transforms + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = test_transforms + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_training)
    if transforms_testing is None:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_testing)
    return DatasetSetup(dataset_name, cifar100_train, cifar100_test)

""" ImageNet """

"""get pytorch preprocessing transforms, version can be 1 or 2"""
def get_pytorch_preprocessing(version=2):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if version == 1:
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return transforms_train, transforms_test
    elif version == 2:
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(),
            TrivialAugmentWide(),  # Equivalent to --auto-augment ta_wide
            RandAugment(num_ops=2, magnitude=9),  # roughly aligns with --randaugment 0.1
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1),
            normalize,
        ])
        transforms_test = transforms.Compose([
            transforms.Resize(232),  # Resize shorter side to 232
            transforms.CenterCrop(224),  # Usually center crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
        return transforms_train, transforms_test
    else:
        raise NotImplementedError


def dataset_imagenet1k(pytorch_preset_version: int, transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet1k' if imagenet1k_path is None else imagenet1k_path
    dataset_name = str(DatasetType.imagenet1k)

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_test = get_pytorch_preprocessing(version=pytorch_preset_version)
    else:
        transforms_train, transforms_test = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet1k_train", transform=transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet1k_test", transform=transforms_test)
    else:
        imagenet_train = datasets.ImageNet(root=dataset_path, split='train', transform=transforms_train)
        imagenet_test = datasets.ImageNet(root=dataset_path, split='val', transform=transforms_test)
    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 1000)))

def dataset_imagenet100(pytorch_preset_version: int, transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet100' if imagenet100_path is None else imagenet100_path
    dataset_name = str(DatasetType.imagenet100)

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_test = get_pytorch_preprocessing(version=pytorch_preset_version)
    else:
        transforms_train, transforms_test = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet100_train", transform = transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet100_test", transform = transforms_test)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 100)))

def dataset_imagenet10(pytorch_preset_version: int, transforms_training=None, transforms_testing=None, enable_memory_cache=False):
    dataset_path = '~/dataset/imagenet10' if imagenet10_path is None else imagenet10_path
    dataset_name = str(DatasetType.imagenet10)

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_test = get_pytorch_preprocessing(version=pytorch_preset_version)
    else:
        transforms_train, transforms_test = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet10_train", transform = transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet10_test", transform = transforms_test)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = transforms_test)

    return DatasetSetup(dataset_name, imagenet_train, imagenet_test, labels=set(range(0, 10)))