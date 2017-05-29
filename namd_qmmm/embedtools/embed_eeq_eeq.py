import numpy as np

from .embed_base import EmbedBase


class EmbedEEqEEq(EmbedBase):

    EMBEDNEAR = 'EEq'
    EMBEDFAR = 'EEq'

    @staticmethod
    def check_unitcell(system):
        if system.n_atoms != system.n_real_qm_atoms + system.n_real_mm_atoms:
            raise ValueError("Unit cell is not complete.")

    def get_near_mask(self):
        return np.array((self.mm_atoms.dij_min <= self.qmCutoff), dtype=bool)

    def get_qm_charge_me(self):
        self.qm_charge_me = np.zeros(self.qm_atoms.n_atoms)

    def check_qm_switching_type(self):
        if self.qmSwitchingType is not None:
            raise ValueError("Switching MM charges is not necessary here.")

    def get_mm_charge(self):

        super(EmbedEEqEEq, self).get_mm_charge()

        self.mm_atoms_near.charge_eeq = self.mm_atoms_near.charge
        self.mm_atoms_far.charge_eeq = self.mm_atoms_far.orig_charge

    def get_mm_esp(self):

        return self.get_mm_esp_eeq().sum(axis=1)
