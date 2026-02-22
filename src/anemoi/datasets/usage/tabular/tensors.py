# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from ..dataset import Dataset
from ..debug import Node
from ..forwards import Forwards

LOG = logging.getLogger(__name__)


def to_numpy(data: Any) -> Any:
    return data


def to_torch(data: Any) -> Any:
    import torch

    from anemoi.datasets.annotated._torch import AnnotatedTorchTensor
    from anemoi.datasets.annotated.metadata import WindowMetaDataBase

    if hasattr(data, "_anemoi_annotation") and isinstance(data._anemoi_annotation, WindowMetaDataBase):
        return AnnotatedTorchTensor(data, data._anemoi_annotation)

    return torch.tensor(data)


class Tensors(Forwards):

    def __init__(self, forward: Dataset, tensors: str = "numpy") -> None:
        super().__init__(forward)
        self.tensors = tensors
        TENSORS = {
            "numpy": to_numpy,
            "torch": to_torch,
        }
        self.to_tensor = TENSORS.get(tensors)
        if self.to_tensor is None:
            raise ValueError(f"Unsupported tensor type: {tensors}, supported types are: {list(TENSORS.keys())}")

    def __getitem__(self, n):
        return self.to_tensor(super().__getitem__(n))

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"tensors": self.tensors}

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], tensors=self.tensors)
