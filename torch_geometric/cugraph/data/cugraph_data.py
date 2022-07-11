from typing import Any, Union, List
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage

import torch
from torch_geometric.data.data import BaseData, RemoteData, Data
from torch import device as TorchDevice
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.cugraph.data.cudf_storage import CudfEdgeStorage, CudfNodeStorage

import numpy as np
import cupy
import cudf

import cugraph
from cugraph import Graph
from cugraph.experimental import PropertyGraph
from datetime import datetime


class CuGraphData(BaseData, RemoteData):
    reserved_keys = [
        PropertyGraph.vertex_col_name,
        PropertyGraph.src_col_name,
        PropertyGraph.dst_col_name,
        PropertyGraph.type_col_name,
        PropertyGraph.edge_id_col_name,
        PropertyGraph.vertex_id_col_name,
        PropertyGraph.weight_col_name
    ] 
    def __init__(self, graph:Union[Graph,PropertyGraph], device:TorchDevice=TorchDevice('cpu'), node_storage:CudfNodeStorage=None, edge_storage:CudfEdgeStorage=None, **kwargs):
        super().__init__()
        
        is_property_graph = isinstance(graph, PropertyGraph)
        if is_property_graph and graph._EXPERIMENTAL__PropertyGraph__vertex_prop_dataframe.index.name != PropertyGraph.vertex_col_name:
            graph._EXPERIMENTAL__PropertyGraph__vertex_prop_dataframe = graph._EXPERIMENTAL__PropertyGraph__vertex_prop_dataframe.set_index(PropertyGraph.vertex_col_name)

        if node_storage is None:
            self.__node_storage = CudfNodeStorage(
                dataframe=graph._vertex_prop_dataframe if is_property_graph \
                    else cudf.DataFrame(
                        cudf.Series(cupy.arange(graph.number_of_vertices()),
                        name=PropertyGraph.vertex_col_name)
                    ),
                device=device, 
                parent=self, 
                reserved_keys=CuGraphData.reserved_keys,
                vertex_col_name=PropertyGraph.vertex_col_name,
                **kwargs
            )
        else:
            self.__node_storage = node_storage

        if edge_storage is None:
            self.__edge_storage = CudfEdgeStorage(
                dataframe=graph._edge_prop_dataframe if is_property_graph else graph.edgelist_df, 
                device=device, 
                parent=self, 
                reserved_keys=CuGraphData.reserved_keys,
                src_col_name=PropertyGraph.src_col_name if is_property_graph else 'src',
                dst_col_name=PropertyGraph.dst_col_name if is_property_graph else 'dst',
                **kwargs
            )
        else:
            self.__edge_storage = edge_storage

        self.graph = graph
        self.device = device
        self.__extracted_subgraph = None
    
    @property
    def _extracted_subgraph(self) -> cugraph.Graph:
        if self.__extracted_subgraph is None:
            self.__extracted_subgraph = cugraph.Graph(directed=True)
            self.__extracted_subgraph.from_cudf_edgelist(
                self.graph._edge_prop_dataframe.join(cudf.Series(cupy.ones(len(self.graph._edge_prop_dataframe), dtype='float32'), name='weight')),
                source=PropertyGraph.src_col_name,
                destination=PropertyGraph.dst_col_name,
                edge_attr='weight',
                renumber=False,
            )

        return self.__extracted_subgraph

    def to(self, to_device: TorchDevice) -> BaseData:
        return CuGraphData(
            graph=self.graph,
            device=TorchDevice(to_device),
            node_storage=self.__node_storage.to(to_device),
            edge_storage=self.__edge_storage.to(to_device)
        )

    def cuda(self):
        return self.to('cuda')
    
    def cpu(self):
        return self.to('cpu')
    
    def stores_as(self, data: 'CuGraphData'):
        print('store as')
        print(type(data))
        print(data.x.shape)
        print(data.num_nodes)
        print(data.num_edges)
        return self
    
    def neighbor_sample(
            self,
            index: Tensor,
            num_neighbors: Tensor,
            replace: bool,
            directed: bool) -> Any:
        
        start_time = datetime.now()

        if not isinstance(index, Tensor):
            index = Tensor(index).to(torch.long)
        if not isinstance(num_neighbors, Tensor):
            num_neighbors = Tensor(num_neighbors).to(torch.long)
        
        index = index.to(self.device)
        num_neighbors = num_neighbors.to(self.device)

        if self.device == 'cpu':
            index = np.array(index)
            # num_neighbors is required to be on the cpu per cugraph api
            num_neighbors = np.array(num_neighbors)
        else:
            index = cupy.from_dlpack(index.__dlpack__())
            # num_neighbors is required to be on the cpu per cugraph api
            num_neighbors = cupy.from_dlpack(num_neighbors.__dlpack__()).get()

        is_property_graph = isinstance(self.graph, PropertyGraph)

        if is_property_graph:
            # FIXME resolve the renumbering issue with extract_subgraph so it can be used here
            #G = self.graph.extract_subgraph(add_edge_data=False, default_edge_weight=1.0, allow_multi_edges=True)
            G = self._extracted_subgraph
        else:
            if self.graph.is_directed() == directed:
                G = self.graph
            elif directed:
                G = self.graph.to_directed()
            else:
                G = self.graph.to_undirected()

        sampling_start = datetime.now()
        index = cudf.Series(index)
        sampling_results = cugraph.uniform_neighbor_sample(
            G,
            index,
            list(num_neighbors), # conversion required by cugraph api
            replace
        )

        end_time = datetime.now()
        td = end_time - start_time
        print('first half', td.total_seconds())
        print('sampling', (end_time - sampling_start).total_seconds())

        start_time = datetime.now()

        nodes_of_interest = cudf.concat([sampling_results.destinations, sampling_results.sources]).unique()
        noi_tensor = torch.from_dlpack(nodes_of_interest.astype('long').to_dlpack())

        renumber_df = cudf.Series(cudf.RangeIndex(len(nodes_of_interest)), index=nodes_of_interest)
        src = torch.from_dlpack(renumber_df[sampling_results.sources].to_cupy(dtype='long').toDlpack())
        dst = torch.from_dlpack(renumber_df[sampling_results.destinations].to_cupy(dtype='long').toDlpack())

        if is_property_graph:
            iloc_start = datetime.now()

            sampled_y = self.y
            if sampled_y is not None:
                sampled_y = sampled_y[noi_tensor]
            
            sampled_x = self.x[noi_tensor]

            iloc_end = datetime.now()
            print('iloc time:', (iloc_end - iloc_start).total_seconds())

            data = Data(
                x=sampled_x, 
                edge_index=torch.concat([dst,src]).reshape((2,src.shape[0])),
                edge_attr=None,
                y=sampled_y
            )
            
        else:
            data = Data(
                x=None,
                edge_index=torch.concat([dst,src]).reshape((2,src.shape[0])),
                edge_attr=None,
                y=None
            )

        end_time = datetime.now()
        td = end_time - start_time
        print('second half', td.total_seconds())

        return data

    def __remove_internal_columns(self, input_cols, additional_columns_to_remove=[]):
        internal_columns = CuGraphData.reserved_keys + additional_columns_to_remove

        # Create a list of user-visible columns by removing the internals while
        # preserving order
        output_cols = list(input_cols)
        for col_name in internal_columns:
            if col_name in output_cols:
                output_cols.remove(col_name)
        
        return output_cols

    def extract_subgraph(
            self, 
            node: Tensor, 
            edges: Tensor, 
            enumerated_edges: Tensor,
            perm: Tensor) -> Any:
        """
        node: Nodes to extract
        edges: Edges to extract (0: src, 1: dst)
        enumerated_edges: Numbered edges to extract
        """
        raise NotImplementedError
    
    @property
    def stores(self) -> List[BaseStorage]:
        return [self.__node_storage, self.__edge_storage]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self.__node_storage]
    
    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self.__edge_storage]
    
    def __getattr__(self, key:str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        
        try:
            return self.__node_storage[key]
        except AttributeError:
            try:
                return self.__edge_storage[key]
            except AttributeError:
                raise AttributeError(key)

    
    def __setattr__(self, key:str, value:Any):
        self.__dict__[key] = value
    
    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            print(key, 'is not string')
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError()

    def stores_as(self, data: 'CuGraphData'):
        return self
    
    @property
    def is_cuda(self) -> bool:
        return self.device.type == 'cuda'

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0