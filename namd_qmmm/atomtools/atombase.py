import copy
import numpy as np

class AtomBase(object):
    """Base class to hold atoms."""

    def __init__(self, x, y, z, charge, index):

        self._atoms = np.recarray(len(x), dtype=[('position', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
                                                 ('element', 'U2'), ('charge', 'f8'), ('index', 'i8')])
        self._atoms.position.x = x
        self._atoms.position.y = y
        self._atoms.position.z = z
        self._atoms.charge = charge
        self._atoms.index = index

        n_real_atoms = np.count_nonzero(self._atoms.index != -1)
        self._real_indices = np.s_[0:n_real_atoms]
        self._virt_indices = np.s_[n_real_atoms:]

        # Define atom indices and mask
        self._indices = np.s_[:]
        self._atom_mask = None

        self._force = np.zeros((self.n_atoms, 3), dtype=float)

    def __len__(self):

        return len(self.atoms)

    def __getitem__(self, item):

        return self.atoms[item]

    @property
    def atoms(self):
        return self._get_property(self._atoms)

    @property
    def real_atoms(self):
        return self.slice_atoms(self._real_indices)

    @property
    def virt_atoms(self):
        return self.slice_atoms(self._virt_indices)

    @property
    def position(self):
        return self.atoms.position.view((float, 3))

    @property
    def charge(self):
        return self.atoms.charge

    @property
    def index(self):
        return self.atoms.index

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def n_real_atoms(self):
        return len(self.real_atoms)

    @property
    def n_virt_atoms(self):
        return len(self.virt_atoms)

    @property
    def force(self):
        return self._get_property(self._force)

    @force.setter
    def force(self, force):
        self._set_property(self._force, force)

    def _get_property(self, name, indices=None):
        """Get property."""
        if indices is None:
            indices = self._indices

        if self._atom_mask is None:
            return name[indices]
        else:
            return name[indices][self._atom_mask[indices]]

    def _set_property(self, name, value, indices=None):
        """Set property."""
        if indices is None:
            indices = self._indices

        if self._atom_mask is None:
            name[indices] = value
        else:
            name[indices][self._atom_mask[indices]] = value

    def slice_atoms(self, indices):
        """Slice atoms."""

        atoms = copy.copy(self)

        atoms._indices = indices

        return atoms

    def mask_atoms(self, mask):
        """Mask atoms."""

        atoms = copy.copy(self)

        if atoms._atom_mask is None:
            atoms._atom_mask = mask
        else:
            atoms._atom_mask = atoms._atom_mask * mask

        return atoms

