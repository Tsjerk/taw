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


class PBC(NDArrayWithAttributes):
    '''
    The PBC class contains an ndarray of which the last two axes
    have shape k by k, and contain a k-dimensional lattice. The
    PBC class may contain its own inverse and implements 
    properties to give access to the lattice, the inverse and
    the transformation to angles. The latter will always contain
    their own inverses for the reverse transformation.
    '''
    
    _attributes = (
        'inverse'
    )
    
    def __new__(cls, pbc, inverse=None):
        if pbc.shape[-1] == 6:
            pbc = cls.dim2pbc(pbc)
        pbc = pbc.view(cls)
        pbc._inverse = inverse
        return pbc
        
    @classmethod
    def dim2pbc(self, arr: np.ndarray) -> np.ndarray:
        '''
        Convert unit cell definition from PDB CRYST1 format to lattice definition.
        '''
        lengths = arr[..., :3].reshape((-1, 3))
        angles = arr[..., 3:].reshape((-1, 3)) * (np.pi / 180)

        cosa = np.cos(angles)
        sing = np.sin(angles[:, 2])

        pbc = np.zeros((len(arr), 9))
        pbc[:, 0] = lengths[:, 0]
        pbc[:, 3] = lengths[:, 1] * cosa[:, 2]
        pbc[:, 4] = lengths[:, 1] * sing
        pbc[:, 6] = lengths[:, 2] * cosa[:, 1]
        pbc[:, 7] = lengths[:, 2] * (cosa[:, 0] - cosa[:, 1] * cosa[:, 2]) / sing
        pbc[:, 8] = (lengths[:, 2] ** 2 - (pbc[:, 6:8] ** 2).sum(axis=1)) ** 0.5

        return pbc.reshape((*arr.shape[:-1], 3, 3))
    
    @property
    def L(self):
        '''Return the 'naked' lattice vectors'''
        return self.view(np.ndarray)
    
    @property
    def I(self):
        '''Return inverse lattice (mapping to unit cube)'''
        if self._inverse is None:
            self._inverse = np.linalg.inv(self.L)
        return PBC(self._inverse, self.L)
    
    @property
    def A(self):
        '''Return transformation to angles'''
        return PBC(self.I * (2 * np.pi), self.L * (0.5 / np.pi))
    
    def reduce(self):
        '''Perform lattice reduction'''
        ...



