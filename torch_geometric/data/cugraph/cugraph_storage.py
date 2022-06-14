from typing import Any
from typing import List

import torch
from torch import device as TorchDevice
from torch_geometric.typing import ProxyTensor
from torch_geometric.data.storage import GlobalStorage

from gaas_client.client import GaasClient

EDGE_KEYS = ["_SRC_", "_DST_"]
VERTEX_KEYS = ["_VERTEX_ID_"]


class TorchTensorGaasGraphDataProxy(ProxyTensor):
    """
    Implements a partial Torch Tensor interface that forwards requests to a
    GaaS server maintaining the actual data in a graph instance.
    The interface supported consists of only the APIs specific DGL workflows
    need - anything else will raise AttributeError.
    """
    _data_categories = ["vertex", "edge"]

    def __init__(self, 
                 gaas_client: GaasClient, 
                 gaas_graph_id: int, 
                 data_category: str, 
                 device:TorchDevice=TorchDevice('cpu'),
                 property_keys: List[str]=None,
                 transposed: bool=False,
                 dtype: torch.dtype=torch.float32):
        if data_category not in self._data_categories:
            raise ValueError("data_category must be one of "
                             f"{self._data_categories}, got {data_category}")

        if property_keys is None:
            if data_category == 'vertex':
                property_keys = VERTEX_KEYS
            else:
                property_keys = EDGE_KEYS

        self.__client = gaas_client
        self.__graph_id = gaas_graph_id
        self.__category = data_category
        self.__device = device
        self.__property_keys = property_keys
        self.__transposed = transposed
        self.dtype = dtype

    def __getitem__(self, index: int):
        """
        Returns a torch.Tensor containing the edge or vertex data (based on the
        instance's data_category) for index, retrieved from graph data on the
        instance's GaaS server.
        """
        # tensor is a transposed dataframe (tensor[0] is df.iloc[0])
        if isinstance(index, torch.Tensor):
            index = [int(i) for i in index]

        if self.__category == "edge":
            # FIXME find a more efficient way to do this that doesn't transfer so much data
            idx = -1 if self.__transposed else index
            data = self.__client.get_graph_edge_dataframe_rows(
                index_or_indices=idx, graph_id=self.__graph_id,
                property_keys=self.__property_keys)

        else:
            # FIXME find a more efficient way to do this that doesn't transfer so much data
            idx = -1 if self.__transposed else index
            data = self.__client.get_graph_vertex_dataframe_rows(
                index_or_indices=idx, graph_id=self.__graph_id,
                property_keys=self.__property_keys)

        if self.__transposed:
            torch_data = torch.from_numpy(data.T)[index].to(self.device)
        else:
            # FIXME handle non-numeric datatypes
            torch_data = torch.from_numpy(data)

        return torch_data.to(self.dtype).to(self.__device)

    @property
    def shape(self) -> torch.Size:
        num_properties = len(self.__property_keys)

        if self.__category == "edge":
            # Handle Edge properties
            if num_properties == 0:
                return torch.Size(
                    self.__client.get_graph_edge_dataframe_shape(self.__graph_id)
                )

            num_edges = self.__client.get_num_edges(self.__graph_id)
            return torch.Size([len(self.__property_keys), num_edges])
        elif self.__category == "vertex":
            # Handle Vertex properties
            if num_properties == 0:
                return torch.Size(
                    self.__client.get_graph_vertex_dataframe_shape(self.__graph_id)
                )

            num_vertices = self.__client.get_num_vertices(self.__graph_id)
            return torch.Size([num_properties, num_vertices])

        raise AttributeError(f'invalid category {self.__category}')

    @property
    def device(self) -> TorchDevice:
        return self.__device
    
    @property
    def is_cuda(self) -> bool:
        return self.__device._type == 'cuda'

    def to(self, to_device: TorchDevice):
        return TorchTensorGaasGraphDataProxy(
            self.__client, 
            self.__graph_id, 
            self.__category, 
            to_device, 
            property_keys=self.__property_keys, 
            transposed=self.__transposed
        )
    
    def dim(self) -> int:
        return self.shape[0]
    
    def size(self, idx=None) -> Any:
        if idx is None:
            return self.shape
        else:
            return self.shape[idx]


class CuGraphStorage(GlobalStorage):
    def __init__(self, gaas_client: GaasClient, gaas_graph_id: int, device: TorchDevice=TorchDevice('cpu'), parent=None):
        super().__init__(_parent=parent)
        setattr(self, 'gaas_client', gaas_client)
        setattr(self, 'gaas_graph_id', gaas_graph_id)
        setattr(self, 'node_index', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex', device, dtype=torch.long))
        setattr(self, 'edge_index', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'edge', device, transposed=True, dtype=torch.long))
        setattr(self, 'x', TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex', device, dtype=torch.float, property_keys=[]))
    
    
    @property
    def num_nodes(self) -> int:
        return self.gaas_client.get_num_vertices(self.gaas_graph_id)
    
    @property
    def num_node_features(self) -> int:
        return self.gaas_client.get_graph_vertex_dataframe_shape(self.gaas_graph_id)[1]
    
    @property
    def num_edge_features(self) -> int:
        # includes the original src and dst columns w/ original names
        return self.gaas_client.get_graph_edge_dataframe_shape(self.gaas_graph_id)[1]

    @property
    def num_edges(self) -> int:
        return self.gaas_client.get_num_edges(self.gaas_graph_id)

    def is_node_attr(self, key: str) -> bool:
        if key == 'x':
            return True
        return self.gaas_client.is_vertex_property(key, self.gaas_graph_id)

    def is_edge_attr(self, key: str) -> bool:
        return self.gaas_client.is_edge_property(key, self.gaas_graph_id)
    
    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        elif self.gaas_client.is_vertex_property(key, self.gaas_graph_id):
            return TorchTensorGaasGraphDataProxy(
                self.gaas_client,
                self.gaas_graph_id,
                'vertex',
                self.node_index.device,
                [key]
            )
        elif self.gaas_client.is_edge_property(key, self.gaas_graph_id):
            return TorchTensorGaasGraphDataProxy(
                self.gaas_client,
                self.gaas_graph_id,
                'edge',
                self.edge_index.device,
                [key]
            )
        
        raise AttributeError(key)