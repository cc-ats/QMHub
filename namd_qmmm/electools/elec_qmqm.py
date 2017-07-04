import math
import numpy as np

from .ewaldsum import EwaldSum
from .elec_core import elec_core_qmqm as elec_core

from .. import units

class ElecQMQM(object):

    def __init__(self, qm_atoms, cell_basis=None):

        self._qm_position = qm_atoms.position

        if cell_basis is not None:
            self.ewald = EwaldSum(cell_basis)
        else:
            self.ewald = None

        self._rij = None
        self._dij2 = None
        self._dij = None

        self._coulomb_esp = None
        self._coulomb_efield = None
        self._ewald_esp = None
        self._ewald_efield = None

    @property
    def rij(self):
        if self._rij is None:
            self._rij = (self._qm_position[np.newaxis, :, :]
                        - self._qm_position[:, np.newaxis, :])
        return self._rij

    @property
    def dij2(self):
        if self._dij2 is None:
            self._dij2 = np.sum(self.rij**2, axis=2)
        return self._dij2

    @property
    def dij(self):
        if self._dij is None:
            self._dij = np.sqrt(self.dij2)
        return self._dij

    @property
    def coulomb_esp(self):
        if self._coulomb_esp is None:
            self._coulomb_esp = units.KE * elec_core.get_coulomb_esp(self.dij)
        return self._coulomb_esp

    @property
    def coulomb_efield(self):
        if self._coulomb_efield is None:
            self._coulomb_efield = units.KE * elec_core.get_coulomb_efield(self.rij, self.dij)
        return self._coulomb_efield

    @property
    def ewald_esp(self):
        if self._ewald_esp is None:
            if self.ewald is not None:
                self._ewald_esp = (units.KE *
                    (elec_core.get_ewald_real_esp(self.rij, self.ewald.real_lattice, self.ewald.alpha) +
                     elec_core.get_ewald_recip_esp(self.rij, self.ewald.recip_lattice, self.ewald.recip_prefactor) -
                     elec_core.get_ewald_self_esp(self.dij, self.ewald.alpha)))

        return self._ewald_esp

    @property
    def ewald_efield(self):
        if self._ewald_efield is None:
            if self.ewald is not None:
                self._ewald_efield = (units.KE *
                    (elec_core.get_ewald_real_efield(self.rij, self.ewald.real_lattice, self.ewald.alpha) +
                     elec_core.get_ewald_recip_efield(self.rij, self.ewald.recip_lattice, self.ewald.recip_prefactor)))

        return self._ewald_efield
