'''
Mask-related utils
'''
import numpy as np
import upolygon
from typing import List


def draw_poly(mask: np.ndarray, poly: List[int], label: int) -> np.ndarray:
    '''
    Draw polygon on mask with coordinates given in [x,y,x,y...] format
    '''
    return upolygon.draw_polygon(mask, [poly], label)