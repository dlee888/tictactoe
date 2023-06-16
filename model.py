import typing

import torch
import torch.nn.functional as F

# class Model(torch.nn.Module):
#     layers: torch.nn.ModuleList
#     activations: typing.List[typing.Callable]
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.layers = torch.nn.ModuleList([torch.nn.Linear(9, 32),
#                                            torch.nn.Linear(32, 9)])
#         self.activations = [F.sigmoid, F.linear]
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for layer, activation in zip(self.layers, self.activations):
#             x = activation(layer(x))
#         return x


class Model(torch.nn.Module):
    fc1: torch.nn.Linear
    fc2: torch.nn.Linear
    fc3: torch.nn.Linear

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(9, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
