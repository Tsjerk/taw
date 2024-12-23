import numpy as np
from .MultiArray import MultiArray


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
        return np.array(*args, **kwargs).view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in self._attributes:
            setattr(self, attr, getattr(obj, attr, None))


class PositionArray(np.ndarray):
    '''This is an alias for the regular np.ndarray'''
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(cls)


class VectorArray(np.ndarray):
    '''A VectorArray is a subclass of np.ndarray that remains invariant under addition and subtraction. 
    Operations that would typically modify the array instead return a copy or the original array, 
    preserving the original data.'''

    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(cls)

    def __add__(self, other):
        '''Return a copy of the VectorArray.

        The addition operation for VectorArray is overridden to return an unchanged copy 
        of the original array, making the array invariant to additions.

        Args:
            other: The value to "add" to the array, which is ignored.

        Returns:
            VectorArray: A copy of the array.
        '''
        return self.copy()
    
    def __iadd__(self, other):
        '''Return the VectorArray itself without modification.

        The in-place addition operation (+=) is overridden to have no effect 
        on the array, preserving its original state.

        Args:
            other: The value to "add" to the array in-place, which is ignored.

        Returns:
            VectorArray: The original array, unchanged.
        '''
        return self
    
    def __sub__(self, other):
        '''Return a copy of the VectorArray.

        The subtraction operation for VectorArray is overridden to return an unchanged copy 
        of the original array, making the array invariant to subtractions.

        Args:
            other: The value to "subtract" from the array, which is ignored.

        Returns:
            VectorArray: A copy of the array.
        '''
        return self.copy()

    def __isub__(self, other):
        '''Return the VectorArray itself without modification.

        The in-place subtraction operation (-=) is overridden to have no effect 
        on the array, preserving its original state.

        Args:
            other: The value to "subtract" from the array in-place, which is ignored.

        Returns:
            VectorArray: The original array, unchanged.
        '''
        return self


class PBC(VectorArray):
    '''
    The PBC class contains an ndarray of which the last two axes
    have shape k by k, and contain a k-dimensional lattice. The
    PBC class implements properties to give access to the lattice, 
    the inverse and the transformation to angles. The latter will 
    always contain their own inverses for the reverse transformation.
    '''
    def __new__(cls, pbc):
        if pbc.shape[-1] == 6:
            return cls.dim2pbc(pbc)
        return pbc.view(cls)
        
    @classmethod
    def dim2pbc(cls, arr: np.ndarray) -> np.ndarray:
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

        return pbc.view(cls).reshape((*arr.shape[:-1], 3, 3)) 
    
    def reduce(self):
        '''Perform lattice reduction'''
        ...


class CoordinatesMaybeWithPBC(NDArrayWithAttributes):
    '''
    The CoordinatesMaybeWithPBC class is intended to 
    store trajectory data that may be accompanied by 
    the periodic boundary condition (PBC) information. 
    If present, the PBC must share the dimensions of
    the trajectory data, except for the second last,
    which is the number of coordinates in the trajectory
    and the dimensionality in the PBC.
    
    Transformations that scale or rotate the coordinates
    are also applied to the PBC. Translations are only
    applied to coordinates.
    
    The class implements several properties for useful
    transformations, namely to box coordinates and to
    angles. Both of these are reversible.
    
    The class implements several operations on the 
    coordinates using the PBC, including putting all
    coordinates inside the base unit cell at the origin
    or around the origin, and putting all coordinates
    in a compact representation.
    '''
    _attributes = (
        'pbc' # [..., 3, 3] lattice matrices
    )
    
    def __new__(cls, *args, pbc=None, **kwargs):
        coordinates = np.array(*args, **kwargs).view(cls)
        coordinates.pbc = pbc
        return coordinates
        
    def _pbcfunc(self, other, operation):
        out = getattr(super(), operation)(other)
        out.__array_finalize__(self)
        if self.pbc is not None:
            out.pbc = getattr(self.pbc, operation)(other)
        return out
        
    def _pbcifunc(self, other, operation):
        getattr(super(), operation)(other)
        if self.pbc is not None:
            getattr(self.pbc, operation)(other)
        return self
        
    def __matmul__(self, other):
        return self._pbcfunc(other, '__matmul__')
    
    def __mul__(self, other):
        return self._pbcfunc(other, '__mul__')
    
    def __div__(self, other):
        return self._pbcfunc(other, '__div__')
    
    def __imul__(self, other):
        return self._pbcifunc(other, '__imul__')
    
    def __idiv__(self, other):
        return self._pbcifunc(other, '__imul__')
    
    @property
    def X(self):
        '''Return bare coordinates'''
        return self.view(np.ndarray)
    
    @property
    def I(self):
        '''Return box coordinates and inverse of lattice'''
        if self.pbc is None:
            return None
        inv = np.linalg.inv(self.pbc)
        # This sets the pbc to identity
        out = self @ inv
        #out.__array_finalize__(self)
        # We reset the pbc with the inverse to allow
        # the reverse operation.
        out.pbc = inv
        return out
    
    @property
    def A(self):
        '''Return coordinates as box angles'''
        if self.pbc is None:
            return None
        inv = np.linalg.inv(self.pbc) * (2 * np.pi)
        # This sets the pbc to 2\pi \times identity
        out = self @ inv
        #out.__array_finalize__(self)
        # We reset the pbc with the inverse to allow
        # the reverse operation with the .C property
        # (It will work with .I, but that does not set
        # the pbc correct afterwards)
        out.pbc = inv
        return out
    
    @property
    def C(self):
        '''Return Cartesian coordinates from box angles'''
        if self.pbc is None:
            return None
        # self.pbc is 2\pi L^-1, inverse is L / 2\pi 
        out = self.I
        out.pbc *= (2 * np.pi)
        return out
        
    def inbox(self):
        '''Put all particles in triclinic unit cell'''
        if self.pbc is None:
            return None
        B = self.I
        B -= np.floor(B)
        return B.I
    
    def originbox(self):
        '''Put all particles in triclinic unit cell around the origin'''
        B = self.I
        B -= np.floor(B + 0.5)
        return B.I
    
    def hexagonal(self):
        '''
        This routine puts the atoms in a compact,  
        hexagonal representation around the origin.
        '''
        B = self.I.reshape((-1, 3))
        # Shift origin to middle of box
        B += 0.5
        B -= np.floor(B)
        x, y = B[:, :2].X.T
        check = (y < 0.5 - x) | (y > 1.5 - x)
        up = y > x
        halfx = 0.5 * x
        twox = 2 * x
        B[check &  up & (y > 1.25 - halfx), 1] -= 1
        B[check & ~up & (y < 0.25 - halfx), 1] += 1
        B[check &  up & (y < 0.5 - twox), 0] += 1
        B[check & ~up & (y > 2.5 - twox), 0] -= 1
        # Shift middle of hexagon to origin and transform
        B -= 0.5
        return B.reshape(self.shape).I
    
    def dodecahedral(self):
        # Easiest is to have a square base.
        # If the base is hexagonal, rotate,
        # do the stuff and rotate back.
        # No one will notice.
        ...



class Trajectory(MultiArray):
    def __init__(self, coordinates, pbc=None, velocities=None, forces=None):
        self._members = {
            'coordinates': PositionArray, 
            'pbc': VectorArray, 
            'velocities': VectorArray, 
            'forces': VectorArray
        }
        for m, t in self._members.items():
            argument = locals()[m]
            setattr(self, m, argument.view(t) if argument is not None else None)

    def __repr__(self):
        nf, na, nd = self.X.shape
        b = (self.B is None) * 'not '
        v = (self.V is None) * 'not '
        f = (self.F is None) * 'not '
        s = f'Trajectory with {nf} frames, {na} atoms and {nd} dimensions. PBC {b}present. Velocities {v}present. Forces {f}present.'
        return s
    
    @property
    def X(self):
        return self.coordinates
    
    @property
    def V(self):
        return self.velocities
    
    @property
    def F(self):
        return self.forces
    
    @property
    def B(self):
        return self.pbc
    
    @property
    def I(self):
        '''Return box coordinates and inverse of lattice'''
        if self.pbc is None:
            return None
        inv = np.linalg.inv(self.pbc)
        print(inv[0])
        # This sets the pbc to identity
        out = self @ inv
        # We reset the pbc with the inverse to allow
        # the reverse operation.
        setattr(out, 'pbc', inv)
        return out
       
    def inbox(self):
        '''Put all particles in triclinic unit cell'''
        if self.pbc is None:
            return None
        B = self.I
        B -= np.floor(B.X)
        return B.I
    
    def originbox(self):
        '''Put all particles in triclinic unit cell around the origin'''
        B = self.I
        B -= np.floor(B.X + 0.5)
        return B.I
    
    def hexagonal(self):
        '''
        This routine puts the atoms in a compact,  
        hexagonal representation around the origin.
        '''
        B = self.I # .reshape((-1, 3))
        X = B.X
        # Shift origin to middle of box
        X += 0.5
        X -= np.floor(X)
        x, y = X[:, :2].T
        check = (y < 0.5 - x) | (y > 1.5 - x)
        up = y > x
        halfx = 0.5 * x
        twox = 2 * x
        X[check &  up & (y > 1.25 - halfx), 1] -= 1
        X[check & ~up & (y < 0.25 - halfx), 1] += 1
        X[check &  up & (y < 0.5 - twox), 0] += 1
        X[check & ~up & (y > 2.5 - twox), 0] -= 1
        # Shift middle of hexagon to origin and transform
        X -= 0.5
        B.X = X.reshape(self.X.shape)
        return B.I
    
    def dodecahedral(self):
        # Easiest is to have a square base.
        # If the base is hexagonal, rotate,
        # do the stuff and rotate back.
        # No one will notice.
        ...
    