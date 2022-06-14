import torch
from typing import Union
from torch import device as TorchDevice
from torch_geometric.data import Data
from torch_geometric.data.cugraph.cugraph_storage import CuGraphStorage

from gaas_client.client import GaasClient
from gaas_client.defaults import graph_id as DEFAULT_GRAPH_ID

class CuGraphData(Data):
    def __init__(self, gaas_client: GaasClient, graph_id: int=DEFAULT_GRAPH_ID, device=TorchDevice('cpu')):
        super().__init__()
        
        # have to access __dict__ here to ensure the store is a CuGraphStorage
        storage = CuGraphStorage(gaas_client, graph_id, device=device, parent=self)
        print('parent', storage._parent)
        self.__dict__['_store'] = storage
        self.device = device
    
    def to(self, to_device: TorchDevice) -> Data:
        return CuGraphData(
            self.gaas_client,
            self.gaas_graph_id,
            TorchDevice(to_device)
        )

    def cuda(self):
        return self.to('cuda')
    
    def cpu(self):
        return self.to('cpu')
    
    def stores_as(self, data: 'Data'):
        return self