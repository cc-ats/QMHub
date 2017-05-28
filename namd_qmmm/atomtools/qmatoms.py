import numpy as np

from .atombase import AtomBase

class QMAtoms(AtomBase):
    """Class to hold QM atoms."""

    def __init__(self, x, y, z, element, charge, index):

        super(QMAtoms, self).__init__(x, y, z, charge, index)

        self._atoms.element = element

        # Set initial QM energy and charges
        self._qm_energy = 0.0
        self._qm_charge = np.zeros(self.n_atoms)
        self._charge_me = np.zeros(self.n_atoms)

    @property
    def element(self):
        return np.char.capitalize(self.atoms.element)

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
