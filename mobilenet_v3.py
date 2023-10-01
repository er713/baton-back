# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains PyTorch model for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset.
"""

from pathlib import Path
from itertools import chain
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from kenning.core.dataset import Dataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.resource_manager import PathOrURI


class PyTorchMobileNetV3(PyTorchWrapper):

    default_dataset = PetDataset
    # pretrained_model_uri = 'kenning:///models/classification/pytorch_pet_dataset_mobilenetv2.pth'  # noqa: E501
    pretrained_model_uri = None

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        class_count: int = 135,
    ):
        super().__init__(model_path, dataset, from_file)
        self.class_count = class_count
        if hasattr(dataset, "numclasses"):
            self.numclasses = dataset.numclasses
        else:
            self.numclasses = class_count

    @classmethod
    def _get_io_specification(cls, numclasses):
        return {
            "input": [
                {
                    "name": "input.1",
                    "shape": (1, 3, 224, 224),
                    "dtype": "float32",
                }
            ],  # noqa: E501
            "output": [
                {"name": "548", "shape": (1, numclasses), "dtype": "float32"}
            ],  # noqa: E501
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(json_dict["class_count"])

    def get_io_specification_from_model(self):
        return self._get_io_specification(self.numclasses)

    def preprocess_input(self, X):
        if np.ndim(X) == 3:
            X = np.array([X])
        import torch

        return (
            torch.Tensor(np.array(X, dtype=np.float32)).to(self.device)
            # .permute(0, 3, 1, 2)
        )

    def create_model_structure(self):
        from torchvision import models
        import torch

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        for param in self.model.parameters():
            param.requires_grad = False
        weights = self.model.classifier[-1].weight
        weights = weights[
            tuple(
                chain(range(151, 272), range(277, 286), [330, 331, 332, 342])
            ),
            :,
        ]
        layer = torch.nn.Linear(1280, self.numclasses)
        weights = torch.concat(
            (deepcopy(weights), torch.rand((1, 1280)) / 10.0)
        )
        layer.weight = torch.nn.parameter.Parameter(
            weights, requires_grad=True
        )
        self.model.classifier[-1] = layer
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def prepare_model(self):
        import torch

        if self.model_prepared:
            return None
        self.create_model_structure()

        self.model_prepared = True
        self.save_model(self.model_path, export_dict=False)
        self.model = torch.jit.script(self.model.eval())
        self.model.to(self.device)

    def train_model(
        self, batch_size: int, learning_rate: int, epochs: int, logdir: Path
    ):
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from torchvision import transforms

        self.prepare_model()
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(0.25)

        self.dataset.standardize = False

        class TrainDatasetPytorch(TorchDataset):
            def __init__(
                self, inputs, labels, dataset, model, dev, transform=None
            ):
                self.inputs = inputs
                self.labels = labels
                self.transform = transform
                self.dataset = dataset
                self.model = model
                self.device = dev

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                X = self.dataset.prepare_input_samples([self.inputs[idx]])[0]
                y = np.array(self.labels[idx])
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)
                if self.transform:
                    X = self.transform(X)
                return (X, y)

        mean, std = self.dataset.get_input_mean_std()

        traindat = TrainDatasetPytorch(
            Xt,
            Yt,
            self.dataset,
            self,
            self.device,
            transform=transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )

        validdat = TrainDatasetPytorch(
            Xv,
            Yv,
            self.dataset,
            self,
            self.device,
            transform=transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )

        trainloader = torch.utils.data.DataLoader(
            traindat, batch_size=batch_size, num_workers=0, shuffle=True
        )

        validloader = torch.utils.data.DataLoader(
            validdat, batch_size=batch_size, num_workers=0, shuffle=True
        )

        self.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        import torch.optim as optim

        opt = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_acc = 0

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=logdir)

        for epoch in range(epochs):
            self.model.train()
            bar = tqdm(trainloader)
            losssum = torch.zeros(1).to(self.device)
            losscount = 0
            for i, (images, labels) in enumerate(bar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                opt.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                opt.step()

                losssum += loss
                losscount += 1
                bar.set_description(f"train epoch: {epoch:3}")
            writer.add_scalar(
                "Loss/train", losssum.data.cpu().numpy() / losscount, epoch
            )

            self.model.eval()
            with torch.no_grad():
                bar = tqdm(validloader)
                total = 0
                correct = 0
                for (images, labels) in bar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    bar.set_description(f"valid epoch: {epoch:3}")
                acc = 100 * correct / total
                writer.add_scalar("Accuracy/valid", acc, epoch)

                if acc > best_acc:
                    self.save_model(self.model_path)
                    best_acc = acc

        self.save_model(
            self.model_path.with_stem(f"{self.model_path.stem}_final")
        )

        self.dataset.standardize = True

        writer.close()
        self.model.eval()

    def convert_input_to_bytes(self, inputdata):
        data = bytes()
        for inp in inputdata.detach().cpu().numpy():
            data += inp.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata):
        import torch

        result = []
        singleoutputsize = self.numclasses * np.dtype(np.float32).itemsize
        for ind in range(0, len(outputdata), singleoutputsize):
            arr = np.frombuffer(
                outputdata[ind : (ind + singleoutputsize)], dtype=np.float32
            )
            result.append(arr)
        return torch.FloatTensor(result)
