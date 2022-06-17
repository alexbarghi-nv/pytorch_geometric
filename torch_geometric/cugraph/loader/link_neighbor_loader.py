from typing import Optional, Callable, Union
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor

from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.link_neighbor_loader import Dataset

from torch_geometric.cugraph.data import GaasData


class GaasLinkNeighborSampler:
    def __init__(self, data, neg_sampling_ratio: float = 0.0):
        pass
    
    def __call__(self):
        # step 1: create negative examples (don't do this for now)

        # step 2: call 
        pass

class GaasLinkNeighborLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        data: GaasData,
        num_neighbors: NumNeighbors,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        neg_sampling_ratio: float = 0.0,
        transform: Callable = None,
        neighbor_sampler: Optional[GaasLinkNeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler
        self.neg_sampling_ratio = neg_sampling_ratio

        if neighbor_sampler is None:
            self.neighbor_sampler = GaasLinkNeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=None,
                neg_sampling_ratio=self.neg_sampling_ratio,
            )

        super().__init__(Dataset(edge_label_index, edge_label),
                         collate_fn=self.neighbor_sampler, **kwargs)
    
    def transform_fn(self, out: Any) -> GaasData:
        node, row, col, edge, edge_label_index, edge_label = out
        data = filter_data(self.data, node, row, col, edge,
                            self.neighbor_sampler.perm)
        data.edge_label_index = edge_label_index
        if edge_label is not None:
            data.edge_label = edge_label

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'