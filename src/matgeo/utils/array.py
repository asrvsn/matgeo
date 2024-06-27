'''
Array utils
'''
from typing import List, Any

def flatten_list(arr: List[List[Any]]) -> List[Any]:
    '''
    Flatten list of lists
    '''
    return [item for sublist in arr for item in sublist]