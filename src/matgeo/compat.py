"""
Compatibility patches for external libraries with NumPy 2.0+
"""
import numpy as np

def patch_numpy_for_meshio():
    """
    Monkey patch NumPy to restore deprecated aliases for meshio compatibility.
    
    This is a temporary fix for meshio/pygmsh compatibility with NumPy 2.0+
    where np.string_ and np.unicode_ were removed.
    """
    if not hasattr(np, 'string_'):
        np.string_ = np.bytes_
    if not hasattr(np, 'unicode_'):
        np.unicode_ = np.str_

# Apply the patch immediately when this module is imported
patch_numpy_for_meshio()
