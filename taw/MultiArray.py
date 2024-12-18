import numpy as np


methods_to_be__distributed = (

    ## Binary operators
    
    '__add__', '__radd__',        # Addition
    '__sub__', '__rsub__',        # Subtraction
    '__mul__', '__rmul__',        # Multiplication
    '__truediv__', '__rtruediv__', # Division
    '__floordiv__', '__rfloordiv__', # Floor division
    '__mod__', '__rmod__',         # Modulus
    '__pow__', '__rpow__',         # Exponentiation
    '__and__', '__rand__',         # Bitwise AND
    '__or__', '__ror__',           # Bitwise OR
    '__xor__', '__rxor__',         # Bitwise XOR
    '__lshift__', '__rlshift__',   # Left shift
    '__rshift__', '__rrshift__',   # Right shift
    '__matmul__', '__rmatmul__',   # Matrix multiplication
    '__getitem__',

    ## Unary operators
    
    '__neg__',       # Negation (-self)
    '__pos__',       # Unary plus (+self)
    '__abs__',       # Absolute value (abs(self))
    '__invert__',    # Bitwise NOT (~self), only for integers

    ## Numpy ndarray methods

    # Aggregation and Reduction
    'sum', 'mean', 'min', 'max', 'std', 'var', 'prod',
    'cumsum', 'cumprod',
    
    # Shape and Reshape
    'reshape', 'transpose', 'flatten', 'ravel', 'T',
    
    # Logical and Comparison
    'all', 'any', 'argmax', 'argmin', 'nonzero', 'clip',
    
    # Mathematical
    'round', 'astype', 'conjugate', 'real', 'imag'
)


def add_distributed_operator_methods(cls):
    """
    Decorator that adds distributed methods to a class. 

    This function dynamically adds methods from `methods_to_be__distributed` to the
    specified class (`cls`). The added methods enable element-wise operations across 
    all `ndarray` members of an instance of the class, such that the operation is 
    applied to each member independently.

    Args:
        cls (type): The class to which distributed methods will be added.

    Returns:
        type: The class with distributed methods added.
    """
    def make_methodfun(op):
        """
        Creates a function that applies a given operation to all members in a class instance.

        Args:
            op (str): The name of the operation or method to apply.

        Returns:
            function: A function that applies the operation to each `ndarray` 
                      member within an instance's `_members` dictionary and 
                      returns a new instance with the results.
        """
        def methodfun(self, *args, **kwargs):
            return self.__class__(
                **{
                    k: None if v is None else getattr(v, op)(*args, **kwargs) 
                    for k, v in self._members.items()
                }
            )
        return methodfun

        # Try to pull the docstring from the corresponding numpy function/operator
        methodfun.__doc__ = (
            f"Distributed version of `{op}`, applies `{op}` to all member arrays.\n\n" + 
            getattr(np.ndarray, op, '')
        )
        
        return methodfun

    # Add the distributed operator methods to the class
    for method in methods_to_be__distributed:
        setattr(cls, method, make_methodfun(method))
    
    return cls


@add_distributed_operator_methods
class MultiArray:
    """
    A class that holds multiple `ndarray` members and supports distributed operations.

    `MultiArray` stores multiple named `ndarray` objects in `_members` and enables
    operations on the entire set of arrays. When a supported operator or method 
    is called on an instance, it applies the operation to each `ndarray` within 
    `_members` individually.

    Attributes:
        _members (dict): A dictionary of named `ndarray` objects.
    """

    def __init__(self, **kwargs):
        """
        Initializes a MultiArray instance with given named arrays.

        Args:
            **kwargs: Named `ndarray` objects to be stored as members of the instance.
        """
        for k, v in kwargs.items():
            super().__setattr__(k, v)
        self._members = { k: getattr(self, k) for k in kwargs }

    def __repr__(self):
        s = ['MultiArray:']
        for k, v in self._members.items():
            s.append(f'{k} ({type(v)}) =\n{v}')
        return '\n'.join(s)
    
    def __setattr__(self, attr, stuff):
        super().__setattr__(attr, stuff)
        if attr in self._members.keys():
            self._members[attr] = getattr(self, attr)
    
