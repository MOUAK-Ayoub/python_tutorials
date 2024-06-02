from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision import datasets
import torch.utils.data

model_name = 'd121'
pre_status = 1
dropout = 1
lr = 0.00001
trans = 5

batch_size = 32
image_size = 512
crop_size = 256
num_workers = 12
num_classes = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_mean = [0.4334, 0.3015, 0.2108]
_std = [0.1994, 0.1386, 0.096]

if trans == 5:
    train_trans = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.RandomAffine(
            degrees=(-180, 180),
            scale=(0.8889, 1.0),
            shear=(-36, 36)),
        transforms.ColorJitter(contrast=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight



