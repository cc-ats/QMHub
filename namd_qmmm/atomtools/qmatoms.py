import numpy as np

from .atombase import AtomBase


class QMAtoms(AtomBase):
    """Class to hold QM atoms."""

    def __init__(self, x, y, z, element, charge, index):

        super(QMAtoms, self).__init__(x, y, z, charge, index)

        self._atoms.element = element

        # Get pair-wise vectors
        self._rij = (self.position[np.newaxis, :, :]
                     - self.position[:, np.newaxis, :])

        # Get pair-wise distances
        self._dij2 = np.sum(self._rij**2, axis=2)
        self._dij = np.sqrt(self._dij2)

        # Set initial QM energy and charges
        self._qm_energy = 0.0
        self._qm_charge = np.zeros(self.n_atoms)
        self._charge_me = np.zeros(self.n_atoms)

    @property
    def element(self):
        return self.atoms.element

    @element.setter
    def element(self, element):
        self._atoms.element = element

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
    def qm_energy(self):
        return self._qm_energy

    @qm_energy.setter
    def qm_energy(self, energy):
        self._qm_energy = energy

    @property
    def qm_charge(self):
        return self._get_property(self._qm_charge)

    @qm_charge.setter
    def qm_charge(self, charge):
        self._set_property(self._qm_charge, charge)

    @property
    def charge_me(self):
        return self._get_property(self._charge_me)

    @charge_me.setter
    def charge_me(self, charge):
        self._set_property(self._charge_me, charge)
