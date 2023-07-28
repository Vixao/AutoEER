from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'cnn',
    'lstm',
    'scnn',
    #'dgcn',
    'lgg',
    'channelWiseAttention',
    #'multi_dimensional_attention'
    'temporal_att'
    #'spatial_att'
]
