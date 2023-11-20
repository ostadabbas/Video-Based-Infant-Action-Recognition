import pickle
import numpy as np
import matplotlib.pyplot as plt

CONVERSION_ORDER  = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 8, 9, 10, 4, 5, 6]

def get_layout(layout: str) -> None:
    """Initialize the layout of candidates."""

    if layout == 'openpose':
        num_node = 18
        inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                        (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                        (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
        center = 1
    elif layout == 'human3.6m':
        num_node = 17
        inward = [(10, 9), (9, 8), (8,7), (7,0), (13, 12), (12, 11), (11, 8),
                  (16, 15), (15, 14), (14, 8),
                  (13, 12), (12, 11), (11, 8),
                  (3, 2), (2,1), (1,0),
                  (6,5), (5,4), (4,0)]
        center = 0
    elif layout == 'nturgb+d':
        num_node = 25
        neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                            (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                            (17, 1), (18, 17), (19, 18), (20, 19), (22, 8),
                            (23, 8), (24, 12), (25, 12)]
        inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
        center = 21 - 1
    elif layout == 'coco':
        num_node = 17
        inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                        (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                        (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
        center = 0
    elif isinstance(layout, dict):
        num_node = layout['num_node']
        inward = layout['inward']
        center = layout['center']
    else:
        raise ValueError(f'Do Not Exist This Layout: {layout}')
    return num_node, inward, center

with open('./NTU/ntu60_3d.pkl', 'rb') as f:
    ntu60_3d = pickle.load(f)

ntu60_3d['annotations'][0]['keypoint'].shape

def convert(skels):
    P, T, _, D = skels.shape
    converted_skels = np.zeros((P, T, 17, D)).astype(skels.dtype)
    converted_skels = skels[:,:,CONVERSION_ORDER,:]
    converted_skels[...,1]*=-1
    return converted_skels

ntu60_3d_converted = ntu60_3d.copy()
for item in ntu60_3d_converted['annotations']:
    new_kps = convert(item['keypoint'])
    item['keypoint'] = new_kps

with open('./NTU/ntu60_3d_h36m.pkl', 'wb') as f:
    pickle.dump(ntu60_3d_converted, f)