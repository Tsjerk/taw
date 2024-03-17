import MDAnalysis as mda


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
