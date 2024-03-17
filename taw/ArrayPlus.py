import numpy as np


class NDArrayWithAttributes(np.ndarray):
    '''
    The NDArrayWithAttributes is a derivative of the NumPy ndarray
    that lists the attributes that are to be set/copied in the
    __array_finalize__ method. Each child should have an attribute
    `_attributes` that lists the attributes characteristic of the
    class
    '''
    _attributes = ()

    def __new__(cls, *args, **kwargs):
        return np.ndarray(*args, **kwargs).view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in self._attributes:
            setattr(self, attr, getattr(obj, attr, None))



