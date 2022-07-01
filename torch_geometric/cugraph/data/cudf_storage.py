from typing import Any, List
from collections.abc import Mapping

import torch
from torch import Tensor
from torch import device as TorchDevice

from torch_geometric.data.storage import EdgeStorage, NodeStorage

import cudf


class CudfStorage():
    """
        Should be compatible with both cudf and dask_cudf
    """
    def __init__(self, dataframe: cudf.DataFrame=cudf.DataFrame(), device: TorchDevice=TorchDevice('cpu'), parent:Any=None, reserved_keys:list=[], **kwargs):
        self._data = dataframe
        self._parent = parent
        self.__device = device
        self.__reserved_keys = list(reserved_keys)
    
    @property
    def device(self):
        return self.__device
    
    @property
    def _reserved_keys(self):
        return self.__reserved_keys

    def to(self, to_device: TorchDevice):
        return CudfStorage(
            self._data, 
            device=to_device, 
            parent=self._parent(), 
            reserved_keys=self.__reserved_keys
        )
    
    def __getattr__(self, key:str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]

        elif key in self._data:
            t = torch.from_dlpack(self._data[key].to_dlpack())
            if self.__device != t.device:
                t = t.to(self.__device)
            return t

        elif key == 'x':
            all_keys = list(self._data.columns)
            for k in self.__reserved_keys + ['y']:
                if k in list(all_keys):
                    all_keys.remove(k)
            
            t = torch.from_dlpack(self._data[all_keys].to_dlpack()).to(torch.float)
            if self.__device != t.device:
                t = t.to(self.__device)
            return t

        raise AttributeError(key)
    
    def __getitem__(self, key:str) -> Any:
        return getattr(self, key)

    @property
    def shape(self) -> tuple:
        return self._data.shape
    
    @property
    def num_features(self) -> int:
        feature_count = self.shape[1]
        for k in self.__reserved_keys:
            if k in list(self._data.columns):
                feature_count -= 1
        
        return feature_count
    
    def __repr__(self) -> str:
        return f'cudf storage ({self.shape[0]}x{self.shape[1]})'
            

class CudfNodeStorage(CudfStorage, NodeStorage):
    def __init__(self, dataframe: cudf.DataFrame, device: TorchDevice=TorchDevice('cpu'), parent:Any=None, vertex_col_name='v', reserved_keys:list=[], key=None):
        super().__init__(dataframe=dataframe, device=device, parent=parent, reserved_keys=reserved_keys)

        self.__vertex_col_name = vertex_col_name
        self.__key = key

    @property
    def _key(self):
        return self.__key

    @property
    def num_nodes(self):
        return self.shape[0]
    
    @property
    def num_node_features(self) -> int:
        feature_count = self.num_features
        if 'y' in list(self._data.columns):
            feature_count -= 1
        return feature_count
    
    @property
    def node_index(self) -> Tensor:
        return self[self.__vertex_col_name].to(torch.long)
    
    def to(self, to_device: TorchDevice):
        return CudfNodeStorage(
            dataframe=self._data, 
            device=to_device, 
            parent=self._parent(), 
            reserved_keys=self._reserved_keys,
            vertex_col_name=self.__vertex_col_name,
            key=self.__key
        )

    def keys(self, *args):
        key_list = [
            'num_nodes',
            'num_node_features',
            'node_index',
            'x',
            'y',
        ]
        
        for a in args:
            key_list.remove(a)
        
        return key_list

    @property
    def _mapping(self) -> Mapping:
        mapping = {}
        
        for k in self.keys():
            mapping[k] = self[k]
        
        return mapping

    
class CudfEdgeStorage(CudfStorage, EdgeStorage):
    def __init__(self, dataframe: cudf.DataFrame, device: TorchDevice=TorchDevice('cpu'), parent:Any=None, src_col_name='src', dst_col_name='dst', reserved_keys:list=[], key=None):
        super().__init__(dataframe=dataframe, device=device, parent=parent, reserved_keys=reserved_keys)

        self.__src_col_name = src_col_name
        self.__dst_col_name = dst_col_name
        self.__key = key
    
    @property
    def _key(self):
        return self.__key

    @property
    def num_edges(self):
        return self.shape[0]
    
    @property
    def num_edge_features(self) -> int:
        return self.num_features
    
    @property
    def edge_index(self) -> Tensor:
        src = self[self.__src_col_name].to(torch.long)
        dst = self[self.__dst_col_name].to(torch.long)
        assert src.shape[0] == dst.shape[0]

        return torch.concat([src,dst]).reshape((2,src.shape[0]))
    
    def to(self, to_device: TorchDevice):
        return CudfEdgeStorage(
            dataframe=self._data, 
            device=to_device, 
            parent=self._parent(), 
            reserved_keys=self._reserved_keys,
            src_col_name=self.__src_col_name,
            dst_col_name=self.__dst_col_name,
            key=self.__key
        )

    def keys(self, *args):
        key_list = [
            'num_edges',
            'num_edge_features',
            'edge_index',
        ]
        
        for a in args:
            key_list.remove(a)
        
        return key_list

    @property
    def _mapping(self) -> Mapping:
        mapping = {}
        
        for k in self.keys():
            mapping[k] = self[k]
        
        return mapping
