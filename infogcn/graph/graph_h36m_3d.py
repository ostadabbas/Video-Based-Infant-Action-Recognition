import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(10, 9), (9, 8), (8,7), (7,0), (13, 12), (12, 11), (11, 8),
                    (16, 15), (15, 14), (14, 8),
                    (13, 12), (12, 11), (11, 8),
                    (3, 2), (2,1), (1,0),
                    (6,5), (5,4), (4,0)]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)

        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
