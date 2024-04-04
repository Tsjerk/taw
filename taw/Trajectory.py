import numpy as np
import MDAnalysis as mda
import struct
import molly
from .ArrayPlus import PBC, CoordinatesMaybeWithPBC


def align_z(v):
    '''
    Generate rotation matrix aligning z-axis onto v (and vice-versa).
    '''
    
    # This routine is based on the notion that for any two
    # vectors, the alignment is a 180 degree rotation 
    # around the resultant vector.
    
    w = v / (v ** 2).sum() ** 0.5
    
    if w[2] <= 1e-8 - 1:
        return -np.eye(3) # Mind the inversion ...

    w[2] += 1
    
    return (2 / (w**2).sum()) * w * w[:, None] - np.eye(3)


class Atoms:
    '''
    This is a wrapper around the MDAnalysis atomgroup that
    updates its atoms when making selections, such that the
    indices are always identical between the atomgroup and
    the corresponding coordinates.
    
    The indices can be used to make a selection from the
    original coordinates.
    '''
    def __init__(self, top, trj=None):
        if isinstance(top, mda.AtomGroup):
            self.universe = mda.Merge(top)
            self.ix = top.ix
            return
        
        if trj is not None:
            self.universe = mda.Universe(top, trj)
        elif isinstance(top, str) and top.lower().endswith('tpr'):
            top = mda.topology.TPRParser.TPRParser(top).parse()
            self.universe = mda.Universe(top)
        self.ix = self.universe.atoms.ix            

    def __getitem__(self, item):
#         match type(item):
#             case int | slice:
#                 # Slice over atoms
#                 ag = self.universe.atoms[item]
#             case str:
#                 ag = self.universe.select_atoms(item)
#             case mda.AtomGroup:
#                 ag = self.universe.atoms & item
#             case _:
#                 raise TypeError(f'Unknown selection type for atomgroup: {item} ({type(item)})')
#         return Atoms(ag)
        if isinstance(item, (int, slice)):
            # Slice over atoms
            ag = self.universe.atoms[item]
        elif isinstance(item, str):
            ag = self.universe.select_atoms(item)
        elif isinstance(item, mda.AtomGroup):
            ag = self.universe.atoms & item
        else:
            raise TypeError(f'Unknown selection type for atomgroup: {item} ({type(item)})')
        return Atoms(ag)


class Trajectory(CoordinatesMaybeWithPBC):
    '''
    The Trajectory class is a derivate of the CoordinatesMaybeWithPBC
    that includes the atoms metadata, through an Atoms class
    property .atoms
    '''
    _attributes = (
        'topfile', # The file from which the topology is read
        'trjfile', # The file from which the trajectory is read
        'atoms', # The atomgroup
        'times', # The times of the frames
        'pbc', # The PBC lattice matrices for all frames
        'centers', # The centers 
        'orientations', # The orientations
        'rgyr_' # Radii of gyration (set by align)
        'rmsd_' # Root mean square deviation (set by align)
    )
    
    def __new__(cls, top: str, trj: str, selection=None, frames=None):

        if trj and trj.endswith(('xtc', 'XTC')):
            # return the XTC version instead
            return XTCTrajectory(top, trj, selection, frames)
        
        # Bookkeeping: MDA stuff
        atomgroup = mda.Universe(top, trj).atoms
        trajectory = atomgroup.universe.trajectory        
        if frames is not None:
            trajectory = trajectory[frames]
        if selection:
            atomgroup = atomgroup.universe.select_atoms(selection)
        natoms = len(atomgroup)
        
        # Setting up trajectory object
        trj = np.empty((len(trajectory), natoms, 3)).view(cls)
        trj.pbc = np.empty((len(trajectory), 6))
        trj.times = np.empty(len(trajectory))
        trj.atoms = Atoms(atomgroup)
        trj.centers = np.zeros((len(trj), 3))
        trj.orientations = np.outer(np.ones(len(trj)), np.eye(3)).reshape((-1, 3, 3))
        trj.topfile = top
        trj.trjfile = trj
            
        # Content: times, pbc, coordinates
        for fidx, frame in enumerate(trajectory):
            trj[fidx] = atomgroup.positions.copy()
            trj.pbc[fidx] = atomgroup.dimensions.copy()
            trj.times[fidx] = frame.time
                
        return trj
       
    def __getitem__(self, item):
        if item is None:
            # Override the behaviour of adding an axis
            # to allow NoneType selections
            # If a newaxis is required as first, then
            # simply also specify the second index
            return self
        if isinstance(item, (str, mda.AtomGroup)):
            # Select a subset of atoms
            atoms = self.atoms[item]
            result = self[..., atoms.ix, :]
            result.atoms = atoms
            return result
        result = super().__getitem__(item)
        if isinstance(item, (int, slice)):
            result.pbc = self.pbc[item]
        elif isinstance(result, type(self)):
            # A bit unfortunate - need to trim the item
            # to have only the frames selection
            # We do neglect the case where dimensions are subset...
            result.pbc = self.pbc[item[:len(self.shape) - 2]]
        return result
        
    def align(self, selection=None, reference=None, around=None):
        '''Align the trajectory with respect to reference and selection'''
        # PBC safe centering
        self.origin(selection)

        fit = self[selection].X
        natoms = fit.shape[-2]
        
        if reference is None:
            # Select first frame, also if shape is complex
            # (multiple trajectories)
            reference = fit.reshape((-1, natoms, fit.shape[-1]))[0]

        # Radii of gyration (for RMSD)
        rgyr2 = (fit ** 2).sum(axis=(-1, -2)) / natoms
        refrg2 = (reference ** 2).sum() / natoms
        
        # Fitting (check for in-plane fitting (any plane))
        if around is not None:
            Z = align_z(np.array(around))
            reference = reference @ Z
            fit = fit @ Z
            U, L, V = np.linalg.svd(reference[:, :2].T @ fit[..., :2])
            dim = slice(None, 2)
        else:
            U, L, V = np.linalg.svd(reference.T @ fit)
            dim = slice(None)
        
        # Rotation matrices
        R = np.zeros_like(self.pbc)
        R[..., dim, dim] = U @ V
        if around is not None:
            # Also rotate vector back from z
            R[..., 2, 2] = 1
            R = Z @ R @ Z.T
        result = self @ R.transpose((0, 2, 1))
        
        # Bookkeeping
        result.orientations = R
        rmsd2 = rgyr2 + refrg2 - 2 * L.sum(axis=-1) / natoms
        rmsd2[rmsd2 < 0] = 0
        result.rmsd_ = rmsd2 ** 0.5
        result.rgyr_ = rgyr2 ** 0.5
        
        return result
        
    def alignxy(self, selection=None, reference=None):
        '''Align the trajectory with respect to reference and selection'''
        return self.align(selection, reference, [0, 0, 1])
        
    def origin(self, selection=None):
        '''Center the selection at the origin for all frames (PBC safe)'''
        if selection is None:
            selection = 'all'
        # Use box coordinate angles to unambiguously define center of mass
        angles = self[selection].A
        cosa, sina = np.cos(angles), np.sin(angles)
        centers = np.arctan2(sina.mean(axis=1), cosa.mean(axis=1))
        centers = centers[:, None] 
        centers = centers @ (0.5 / np.pi * self.pbc)
        self.centers = centers
        self -= centers
        return self

    def center(self, selection=None):
        '''Center the selection in the center of the triclinic cell (PBC safe'''
        return self.origin(selection) + 0.5 * self.pbc.sum(axis=1)[:, None]
        
    def molbox(self, selection):
        # This may be expensive...
        # Voxelized will be faster
        ...
        
    def split(self, what):
        '''Split the trajectory according to mda.atomgroup.split'''
        return [ self[ag] for ag in self.atoms.split(what) ]


class XTCTrajectory(Trajectory):
    '''
    The XTC Trajectory class derives from the Trajectory, but
    reads in its coordinates efficiently using `molly`.
    '''
    
    def __new__(cls, top: str, trj: str, selection=None, frames=None):

        # Bookkeeping: MDA stuff
        atomgroup = mda.Universe(top, trj).atoms
        if selection:
            atomgroup = atomgroup.universe.select_atoms(selection)
        natoms = len(atomgroup)

        # Indexing XTC trajectory
        with open(trj, 'rb') as xtc:
            filesize = xtc.seek(0, 2)
            positions = [ xtc.seek(0) ]
            while positions[-1] < filesize:
                xtc.seek(88, 1)
                framesize = struct.unpack('>l', xtc.read(4))[0]
                framesize += -framesize % 4
                positions.append(xtc.seek(framesize, 1))
        positions = np.array(positions[:-1])

        if frames is not None:
            positions = positions[frames]

        # Allocating memory, setting up structure
        xtc = np.empty((len(positions), natoms, 3), dtype=np.float32).view(cls)
        xtc.pbc = np.empty((len(positions), 3, 3), dtype=np.float32)
        xtc.times = np.empty(len(positions), dtype=np.float32)
        xtc.atoms = Atoms(atomgroup)
        xtc.centers = np.zeros((len(trj), 3))
        xtc.orientations = np.outer(np.ones(len(trj)), np.eye(3)).reshape((-1, 3, 3))
        xtc.topfile = top
        xtc.trjfile = trj
            
        # Fill content: times, pbc, coordinates
        X = molly.XTCReader(trj)
        X.read_into_array(xtc, xtc.pbc, xtc.times, frames, atomgroup.ix.tolist())

        # To be compatible with the Trajectory class that uses MDAnalysis we need
        # to convert from nm to Å.
        xtc *= 10
        
        return xtc


