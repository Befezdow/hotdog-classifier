import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms


class ScoreImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ScoreImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class DataManager:
    def __init__(self, train_path='./data/train', test_path='./data/test', batch_size=32, transform_image_size=224):
        self._test_path = test_path
        self._train_path = train_path
        self._batch_size = batch_size
        self._transform_image_size = transform_image_size

        self._net_transform = transforms.Compose([
            transforms.Resize(255), transforms.CenterCrop(self._transform_image_size), transforms.ToTensor()
        ])

        self._svm_transform = transforms.Compose([
            transforms.Resize(255), transforms.CenterCrop(self._transform_image_size), transforms.ToTensor()
        ])

        self.cuda_params = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

    def get_net_train_test_loaders(self, test_part=0.2, need_shuffle=True):
        dataset = datasets.ImageFolder(self._train_path, transform=self._net_transform)

        dataset_size = len(dataset)
        test_size = int(dataset_size * test_part)
        train_size = dataset_size - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )

        return train_loader, test_loader

    def get_net_score_loader(self):
        dataset = ScoreImageFolder(self._test_path, transform=self._net_transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, **self.cuda_params)

    def get_svm_train_test_data(self, test_part=0.2):
        def _flatten_dataset(_dataset, _dataset_size):
            size = self._transform_image_size
            x = np.zeros((_dataset_size, size * size * 3), dtype=np.float32)
            y = np.zeros(_dataset_size)
            for idx, (image, label) in enumerate(_dataset):
                x[idx][:] = image.flatten()
                y[idx] = label
            return x, y

        dataset = datasets.ImageFolder(self._train_path, transform=self._svm_transform)

        dataset_size = len(dataset)
        test_size = int(dataset_size * test_part)
        train_size = dataset_size - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        x_train, y_train = _flatten_dataset(train_dataset, train_size)
        x_test, y_test = _flatten_dataset(test_dataset, test_size)

        return (x_train, y_train), (x_test, y_test)

    def get_svm_score_data(self):
        def _flatten_dataset(_dataset, _dataset_size):
            size = self._transform_image_size
            x = np.zeros((_dataset_size, size * size * 3), dtype=np.float32)
            file_names = []
            for idx, (image, _, file_name) in enumerate(_dataset):
                x[idx][:] = image.flatten()
                file_names.append(file_name)
            return x, file_names

        dataset = ScoreImageFolder(self._test_path, transform=self._svm_transform)
        dataset_size = len(dataset)

        return _flatten_dataset(dataset, dataset_size)