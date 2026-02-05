import os, sys, json
from enum import Enum, auto
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import TrivialAugmentWide
from torchvision.transforms.v2 import RandAugment

from .dataset_masked import MaskedImageDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src.util import expand_path
from py_src.ml_setup_base.dataset_intermediate_layer import DatasetWithCachedOutputInSharedMem, DatasetWithCachedOutputInMem, ImageDatasetWithCachedInputInSharedMem
from py_src.ml_setup_base.base import DatasetSetup
from py_src.torch_vision_train import presets

default_path_mnist = expand_path('~/dataset/mnist')
default_path_cifar10 = expand_path('~/dataset/cifar10')
default_path_cifar100 = expand_path('~/dataset/cifar100')
default_path_svhn = expand_path('~/dataset/svhn')
default_path_imagenet1k = expand_path('~/dataset/imagenet1k')
default_path_imagenet100 = expand_path('~/dataset/imagenet100')
default_path_imagenet10 = expand_path('~/dataset/imagenet10')
default_path_flickr30k = expand_path('~/dataset/flickr30k')

default_path_random_mnist = expand_path('~/dataset/random_mnist')
default_path_random_cifar10 = expand_path('~/dataset/random_cifar10')
default_path_random_cifar100 = expand_path('~/dataset/random_cifar100')
default_path_random_imagenet10 = expand_path('~/dataset/random_imagenet10')
default_path_random_imagenet100 = expand_path('~/dataset/random_imagenet100')
default_path_random_imagenet1k = expand_path('~/dataset/random_imagenet1k')

""" Load env override file """
imagenet1k_path = None
imagenet100_path = None
imagenet10_path = None
flickr30k_path = None

# load override dataset path from dataset_env.py
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
    if hasattr(env, "flickr30k_path"):
        flickr30k_path = env.flickr30k_path
if imagenet1k_path is None:
    imagenet1k_path = default_path_imagenet1k
if imagenet100_path is None:
    imagenet100_path = default_path_imagenet100
if imagenet10_path is None:
    imagenet10_path = default_path_imagenet10
if flickr30k_path is None:
    flickr30k_path = default_path_flickr30k

""" Dataset Enum """
class DatasetType(Enum):
    default = auto()
    mnist = auto()
    mnist_224 = auto()
    random_mnist = auto()
    cifar10 = auto()
    cifar10_224 = auto()
    random_cifar10 = auto()
    cifar100 = auto()
    cifar100_224 = auto()
    random_cifar100 = auto()
    imagenet10 = auto()
    random_imagenet10 = auto()
    imagenet100 = auto()
    random_imagenet100 = auto()
    imagenet1k = auto()
    random_imagenet1k = auto()
    imagenet1k_sam_mask_random_noise = auto()
    imagenet1k_sam_mask_black = auto()
    svhn = auto()
    flickr30k = auto()

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

def dataset_type_to_random(in_type: DatasetType) -> DatasetType:
    if in_type == DatasetType.mnist:
        return DatasetType.random_mnist
    elif in_type == DatasetType.cifar10:
        return DatasetType.random_cifar10
    elif in_type == DatasetType.cifar100:
        return DatasetType.random_cifar100
    elif in_type == DatasetType.imagenet10:
        return DatasetType.random_imagenet10
    elif in_type == DatasetType.imagenet100:
        return DatasetType.random_imagenet100
    elif in_type == DatasetType.imagenet1k:
        return DatasetType.random_imagenet1k
    else:
        raise NotImplementedError


""" MNIST """
def dataset_mnist(rescale_to_224=False, random_rotation=5, override_dataset_path=None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_mnist if override_dataset_path is None else override_dataset_path
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True)
    mean = mnist_train.data.float().mean() / 255
    std = mnist_train.data.float().std() / 255
    if rescale_to_224:
        dataset_type = DatasetType.mnist_224
    else:
        dataset_type = DatasetType.mnist
    dataset_name = str(dataset_type.name)

    train_transforms = []
    test_transforms = []
    # data augmentation
    if augmentation:
        if random_rotation != 0:
            train_transforms.append(transforms.RandomRotation(random_rotation, fill=(0,)))
    if rescale_to_224:
        train_transforms.append(transforms.Resize((224, 224)))
        test_transforms.append(transforms.Resize((224, 224)))
    else:
        train_transforms.append(transforms.RandomCrop(28, padding=2))
    train_transforms = train_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    test_transforms = test_transforms + [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
    train_data = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transforms.Compose(train_transforms))
    test_data = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transforms.Compose(test_transforms))
    return DatasetSetup(dataset_name, dataset_type, train_data, test_data, labels=set(range(10)))

""" CIFAR10 """
def dataset_cifar10(rescale_to_224=False, transforms_training=None, transforms_testing=None, mean_std=None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_cifar10
    if rescale_to_224:
        dataset_type = DatasetType.cifar10_224
    else:
        dataset_type = DatasetType.cifar10
    dataset_name = str(dataset_type.name)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))

    # data augmentation
    if augmentation:
        data_augmentation = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
    else:
        data_augmentation = []
    if rescale_to_224:
        transform_rescale_to_224 = transforms.Resize((224, 224))
    else:
        transform_rescale_to_224 = []
    train_transforms = data_augmentation + transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms_training)
    if transforms_testing is None:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar10_test = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms_testing)

    return DatasetSetup(dataset_name, dataset_type, cifar10_train, cifar10_test, labels=set(range(10)))



""" CIFAR100 """
def dataset_cifar100(rescale_to_224=False, transforms_training=None, transforms_testing=None, mean_std=None, augmentation=True, *args, **kwargs):
    dataset_path = default_path_cifar100
    if rescale_to_224:
        dataset_type = DatasetType.cifar100_224
    else:
        dataset_type = DatasetType.cifar100
    dataset_name = str(dataset_type.name)
    if mean_std is not None:
        stats = mean_std
    else:
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # data augmentation
    if augmentation:
        data_augmentation = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(32, padding=4, padding_mode='reflect')]
    else:
        data_augmentation = []
    if rescale_to_224:
        transform_rescale_to_224 = transforms.Resize((224, 224))
    else:
        transform_rescale_to_224 = []
    train_transforms = data_augmentation + transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]
    test_transforms = transform_rescale_to_224 + [transforms.ToTensor(), transforms.Normalize(*stats)]

    if transforms_training is None:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.Compose(train_transforms))
    else:
        cifar100_train = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms_training)
    if transforms_testing is None:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms.Compose(test_transforms))
    else:
        cifar100_test = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transforms_testing)
    return DatasetSetup(dataset_name, dataset_type, cifar100_train, cifar100_test, labels=set(range(0, 100)))

""" SVHN """
def dataset_svhn(transforms_training=None, transforms_testing=None, mean_std=None, use_extra=False, augmentation=True, *args, **kwargs):
    dataset_path = default_path_svhn
    if mean_std is None:
        mean_std = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    dataset_type = DatasetType.svhn
    dataset_name = DatasetType.svhn.name

    if augmentation:
        data_augmentation = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.5),]
    else:
        data_augmentation = [transforms.Resize(32)]

    if transforms_training is None:
        transforms_training = transforms.Compose(data_augmentation + [
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
        ])
    svhn_train = datasets.SVHN(root=dataset_path, split='train', download=True, transform=transforms_training)
    if transforms_testing is None:
        transforms_testing = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
        ])
    svhn_test = datasets.SVHN(root=dataset_path, split='test', download=True, transform=transforms_testing)
    if use_extra:
        extra = datasets.SVHN(root=dataset_path, split='extra', download=True, transform=transforms_training)
        svhn_train = ConcatDataset([svhn_train, extra])
    return DatasetSetup(dataset_name, dataset_type, svhn_train, svhn_test, labels=set(range(0, 10)))

""" ImageNet """

"""get pytorch preprocessing transforms, version can be 1 or 2"""
def get_pytorch_preprocessing(version=2, train_crop_size=None, val_resize_size=None, val_crop_size=None, random_erasing=None, augmentation=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not augmentation:
        train_crop_size = 224 if train_crop_size is None else train_crop_size
        val_resize_size = 256 if val_resize_size is None else val_resize_size
        val_crop_size = 224 if val_crop_size is None else val_crop_size
        transforms_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(train_crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        transforms_test = transforms.Compose([
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        return transforms_train, transforms_test
    if version == 1:
        train_crop_size = 224 if train_crop_size is None else train_crop_size
        val_resize_size = 256 if val_resize_size is None else val_resize_size
        val_crop_size = 224 if val_crop_size is None else val_crop_size
        transforms_train = [transforms.RandomResizedCrop(train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]
        if random_erasing is not None:
            transforms_train.append(transforms.RandomErasing(random_erasing))
        transforms_train = transforms.Compose(transforms_train)
        transforms_test = transforms.Compose([
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        return transforms_train, transforms_test
    elif version == 2:
        train_crop_size = 176 if train_crop_size is None else train_crop_size
        val_resize_size = 232 if val_resize_size is None else val_resize_size
        val_crop_size = 224 if val_crop_size is None else val_crop_size
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(train_crop_size),
            transforms.RandomHorizontalFlip(),
            TrivialAugmentWide(),  # Equivalent to --auto-augment ta_wide
            RandAugment(num_ops=2, magnitude=9),  # roughly aligns with --randaugment 0.1
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1 if random_erasing is None else random_erasing),
            normalize,
        ])
        transforms_test = transforms.Compose([
            transforms.Resize(val_resize_size),  # Resize shorter side to 232
            transforms.CenterCrop(val_crop_size),  # Usually center crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
        return transforms_train, transforms_test
    else:
        raise NotImplementedError

def dataset_imagenet1k(pytorch_preset_version: int, transforms_training=None, transforms_testing=None,
                       train_crop_size=None, val_resize_size=None, val_crop_size=None,
                       random_erasing=None, enable_memory_cache=False, augmentation=True,
                       *args, **kwargs):
    dataset_name = str(DatasetType.imagenet1k.name)
    dataset_type = DatasetType.imagenet1k

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_val = get_pytorch_preprocessing(version=pytorch_preset_version, train_crop_size=train_crop_size,
                                                                      val_resize_size=val_resize_size, val_crop_size=val_crop_size,
                                                                      random_erasing=random_erasing, augmentation=augmentation)
    else:
        transforms_train, transforms_val = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(imagenet1k_path, "train"), "imagenet1k_train", transform=transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(imagenet1k_path, "val"), "imagenet1k_test", transform=transforms_val)
    else:
        imagenet_train = datasets.ImageNet(root=imagenet1k_path, split='train', transform=transforms_train)
        imagenet_test = datasets.ImageNet(root=imagenet1k_path, split='val', transform=transforms_val)
    return DatasetSetup(dataset_name, dataset_type, imagenet_train, imagenet_test, labels=set(range(0, 1000)))

def dataset_imagenet100(pytorch_preset_version: int, transforms_training=None, transforms_testing=None,
                        train_crop_size=None, val_resize_size=None, val_crop_size=None,
                        random_erasing=None, enable_memory_cache=False, augmentation=True,
                        *args, **kwargs):
    dataset_path = default_path_imagenet100 if imagenet100_path is None else imagenet100_path
    dataset_name = str(DatasetType.imagenet100.name)
    dataset_type = DatasetType.imagenet100

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_val = get_pytorch_preprocessing(version=pytorch_preset_version, train_crop_size=train_crop_size,
                                                                      val_resize_size=val_resize_size, val_crop_size=val_crop_size,
                                                                      random_erasing=random_erasing, augmentation=augmentation)
    else:
        transforms_train, transforms_val = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet100_train", transform = transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet100_test", transform = transforms_val)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = transforms_val)

    return DatasetSetup(dataset_name, dataset_type, imagenet_train, imagenet_test, labels=set(range(0, 100)))

def dataset_imagenet10(pytorch_preset_version: int, transforms_training=None, transforms_testing=None,
                       train_crop_size=None, val_resize_size=None, val_crop_size=None,
                       random_erasing=None, enable_memory_cache=False, augmentation=True,
                       *args, **kwargs):
    dataset_path = default_path_imagenet10 if imagenet10_path is None else imagenet10_path
    dataset_name = str(DatasetType.imagenet10.name)
    dataset_type = DatasetType.imagenet10

    if transforms_testing is None and transforms_training is None:
        transforms_train, transforms_val = get_pytorch_preprocessing(version=pytorch_preset_version, train_crop_size=train_crop_size,
                                                                      val_resize_size=val_resize_size, val_crop_size=val_crop_size,
                                                                      random_erasing=random_erasing, augmentation=augmentation)
    else:
        transforms_train, transforms_val = transforms_training, transforms_testing

    if enable_memory_cache:
        imagenet_train = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "train"), "imagenet10_train", transform = transforms_train)
        imagenet_test = ImageDatasetWithCachedInputInSharedMem(os.path.join(dataset_path, "val"), "imagenet10_test", transform = transforms_val)
    else:
        imagenet_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform = transforms_train)
        imagenet_test = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform = transforms_val)

    return DatasetSetup(dataset_name, dataset_type, imagenet_train, imagenet_test, labels=set(range(0, 10)))


def dataset_imagenet1k_custom(train_crop_size=224, val_resize_size=256, val_crop_size=224,
                              interpolation=transforms.InterpolationMode.BILINEAR, auto_augment_policy=None,
                              random_erase_prob=0.0, ra_magnitude=9, augmix_severity=3,
                              backend='pil', use_v2=False):
    dataset_name = str(DatasetType.imagenet1k.name)
    dataset_type = DatasetType.imagenet1k
    dataset_path = f'{default_path_imagenet1k}/train' if imagenet1k_path is None else f"{imagenet1k_path}/train"
    dataset_train = datasets.ImageFolder(
        dataset_path,
        presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
            backend=backend,
            use_v2=use_v2,
        ),
    )
    dataset_path = f'{default_path_imagenet1k}/val' if imagenet1k_path is None else f"{imagenet1k_path}/val"
    transforms_test = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_test = datasets.ImageFolder(dataset_path, transforms_test)
    return DatasetSetup(dataset_name, dataset_type, dataset_train, dataset_test, labels=set(range(0, 1000)))


def dataset_imagenet1k_sam_mask_random_noise(train_crop_size=224, val_resize_size=256, val_crop_size=224,
                                return_path=False, *args, **kwargs):
    dataset_name = str(DatasetType.imagenet1k_sam_mask_random_noise.name)
    dataset_type = DatasetType.imagenet1k_sam_mask_random_noise
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = MaskedImageDataset(image_root=os.path.join(imagenet1k_path, "train"),
                                       mask_root=os.path.join(imagenet1k_path, "train_sam_mask"),
                                       transform=transforms_train, return_paths=return_path,
                                       unmasked_area_type="random", use_imagenet_label=True)

    transforms_test = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_test = datasets.ImageNet(root=imagenet1k_path, split='val', transform=transforms_test)
    output = DatasetSetup(dataset_name, dataset_type, dataset_train, dataset_test, labels=set(range(0, 1000)))
    output.is_masked_dataset = True
    return output


def dataset_imagenet1k_sam_mask_black(train_crop_size=224, val_resize_size=256, val_crop_size=224,
                                return_path=False, *args, **kwargs):
    dataset_name = str(DatasetType.imagenet1k_sam_mask_black.name)
    dataset_type = DatasetType.imagenet1k_sam_mask_black
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(train_crop_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = MaskedImageDataset(image_root=expand_path('~/dataset/imagenet1k/train'),
                                       mask_root=expand_path('~/dataset/imagenet1k/train_sam_mask'),
                                       transform=transforms_train, return_paths=return_path,
                                       unmasked_area_type="zero", use_imagenet_label=True)

    transforms_test = transforms.Compose([
        transforms.Resize(val_resize_size),
        transforms.CenterCrop(val_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_test = datasets.ImageNet(root=imagenet1k_path, split='val', transform=transforms_test)
    output = DatasetSetup(dataset_name, dataset_type, dataset_train, dataset_test, labels=set(range(0, 1000)))
    output.is_masked_dataset = True
    return output

""" Flickr """
from . import dataset_flickr
def dataset_flickr30k(*args, **kwargs):
    dataset_name = str(DatasetType.flickr30k.name)
    dataset_type = DatasetType.flickr30k

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.), # no hue because it distorts the colors
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset_train = dataset_flickr.Flickr30k(flickr30k_path, split='train', img_transform=train_transform)
    dataset_test = dataset_flickr.Flickr30k(flickr30k_path, split='val', img_transform=valid_transform)
    output = DatasetSetup(dataset_name, dataset_type, dataset_train, dataset_test, labels="non_classifier")
    return output

def get_dataset_random(dataset_type, default_dataset_path, override_dataset_path, channel, label_count):
    dataset_path = default_dataset_path if override_dataset_path is None else override_dataset_path
    dataset_name = str(dataset_type.name)

    transforms_train = []
    transforms_test = []

    mean_std_file_path = os.path.join(dataset_path, "mean_std.json")
    if os.path.exists(mean_std_file_path):
        with open(mean_std_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mean = data["mean"]
        std = data["std"]
    else:
        dataset_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transforms.Compose([transforms.ToTensor()]))
        mean, std = calculate_mean_std(dataset_train)
        mean, std = mean.tolist(), std.tolist()
    if isinstance(mean, (int, float)):
        mean = [float(mean)]
    if isinstance(std, (int, float)):
        std = [float(std)]

    if channel == 1:
        mean = [sum(mean) / len(mean)]
        std = [sum(std) / len(std)]
        transforms_train.append(transforms.Grayscale())
        transforms_test.append(transforms.Grayscale())
    elif channel == 3:
        pass
    else:
        raise NotImplementedError

    transforms_train = transforms.Compose(transforms_train + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    transforms_test = transforms.Compose(transforms_test + [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dataset_train = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transforms_train)
    dataset_test = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transforms_test)
    return DatasetSetup(dataset_name, dataset_type, dataset_train, dataset_test, labels=set(range(label_count)))

""" Random MNIST """
def dataset_random_mnist(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_mnist, default_path_random_mnist, override_dataset_path, 1, 10)

""" Random CIFAR10 """
def dataset_random_cifar10(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_cifar10, default_path_random_cifar10, override_dataset_path, 3, 10)

""" Random CIFAR100 """
def dataset_random_cifar100(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_cifar100, default_path_random_cifar100, override_dataset_path, 3, 100)

""" Random Imagenet """
def dataset_random_imagenet10(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet10, default_path_random_imagenet10, override_dataset_path, 3, 10)

def dataset_random_imagenet100(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet100, default_path_random_imagenet100, override_dataset_path, 3, 100)

def dataset_random_imagenet1k(override_dataset_path=None, *args, **kwargs):
    return get_dataset_random(DatasetType.random_imagenet1k, default_path_random_imagenet1k, override_dataset_path, 3, 1000)

# helper functions
dataset_type_to_setup = {
    DatasetType.mnist: dataset_mnist,
    DatasetType.cifar10: dataset_cifar10,
    DatasetType.cifar100: dataset_cifar100,
    DatasetType.imagenet10: dataset_imagenet10,
    DatasetType.imagenet100: dataset_imagenet100,
    DatasetType.imagenet1k: dataset_imagenet1k,
    DatasetType.imagenet1k_sam_mask_random_noise: dataset_imagenet1k_sam_mask_random_noise,
    DatasetType.imagenet1k_sam_mask_black: dataset_imagenet1k_sam_mask_black,
    DatasetType.flickr30k: dataset_flickr30k,

    DatasetType.random_mnist: dataset_random_mnist,
    DatasetType.random_cifar10: dataset_random_cifar10,
    DatasetType.random_cifar100: dataset_random_cifar100,
    DatasetType.random_imagenet10: dataset_random_imagenet10,
    DatasetType.random_imagenet100: dataset_random_imagenet100,
    DatasetType.random_imagenet1k: dataset_random_imagenet1k,
}
