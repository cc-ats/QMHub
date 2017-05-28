import numpy as np

from .atombase import AtomBase

class MMAtoms(AtomBase):
    """Class to hold MM atoms."""

    def __init__(self, x, y, z, charge, index, orig_charges, qm_atoms):

        super(MMAtoms, self).__init__(x, y, z, charge, index)

        # # Get cell basis
        # self._cell_basis = cell_basis

        # Embed QM atoms
        self._qm_atoms = qm_atoms

        # Get pair-wise vectors
        self._rij = (self._qm_atoms.position[np.newaxis, :, :]
                     - self.position[:, np.newaxis, :])

        # # Apply minimum image convention
        # if self._cell_basis.sum() == 0:
        #     _rij -= np.diagonal(self._cell_basis) * np.rint(_rij / np.diagonal(self._cell_basis))

        # Get pair-wise distances
        self._dij2 = np.sum(self._rij**2, axis=2)
        self._dij = np.sqrt(self._dij2)

        # Get minimum distances between QM and MM atoms
        self._dij_min2 = self._dij2[:, 0:self._qm_atoms.n_real_atoms].min(axis=1)
        self._dij_min2[self._virt_indices] = 0.0
        self._dij_min_j = self._dij2[:, 0:self._qm_atoms.n_real_atoms].argmin(axis=1)
        self._dij_min_j[self._virt_indices] = -1
        self._dij_min = np.sqrt(self._dij_min2)

        # Initialize original MM charges
        self._orig_charges = np.zeros(self.n_atoms, dtype=float)
        self._orig_charges[self._real_indices] = orig_charges

        # Get array mask to cancel 1-2 and 1-3 interactions for coulomb
        self._coulomb_mask = np.ones((self.n_atoms, self._qm_atoms.n_atoms), dtype=bool)

        self._esp_eed = np.zeros(self.n_atoms, dtype=float)

    @property
    def rij(self):
        return self._get_property(self._rij)

    @property
    def dij2(self):
        return self._get_property(self._dij2)

    @property
    def dij(self):
        return self._get_property(self._dij)

    @property
    def dij_min2(self):
        return self._get_property(self._dij_min2)

    @property
    def dij_min(self):
        return self._get_property(self._dij_min)

    @property
    def dij_min_j(self):
        return self._get_property(self._dij_min_j)

    @property
    def orig_charges(self):
        return self._get_property(self._orig_charges, self._real_indices)

    @property
    def coulomb_mask(self):
        return self._get_property(self._coulomb_mask)

    @coulomb_mask.setter
    def coulomb_mask(self, mask):
        self._set_property(self._coulomb_mask, mask)

    @property
    def esp_eed(self):
        return self._get_property(self._esp_eed)

    @esp_eed.setter
    def esp_eed(self, esp):
        self._set_property(self._esp_eed, esp)
