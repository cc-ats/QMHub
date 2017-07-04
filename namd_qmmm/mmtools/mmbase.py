import numpy as np


class MMBase(object):

    MMTOOL = None

    @property
    def qm_energy(self):
        return self.qm_atoms.qm_energy

    @property
    def qm_charge(self):
        return self.qm_atoms.qm_charge

    @property
    def qm_charge_me(self):
        return self.qm_atoms.charge_me

    @property
    def qm_force(self):
        return self.qm_atoms.force

    @property
    def mm_force(self):
        return self.mm_atoms.force

    @property
    def mm_esp_eed(self):
        return self.mm_atoms.esp_eed

    @staticmethod
    def parse_output(qm):
        """Parse the output of QM calculation."""

        if qm.calc_forces:
            qm.parse_output()

    def apply_corrections(self, embed):
        """Correct the results."""

        if self.qm_energy != 0.0:
            self.calc_me(embed)
            self.corr_scaling(embed)
            self.corr_virt_mm_atoms()

    def calc_me(self, embed):
        """Calculate forces and energy for mechanical embedding."""

        if embed.mm_atoms_near.charge_me is not None:

            mm_esp_me = embed.get_mm_esp_me()
            mm_efield_me = embed.get_mm_efield_me()
            mm_charge = embed.mm_atoms_near.charge_me

            energy = mm_charge[:, np.newaxis] * mm_esp_me

            force = -1 * mm_charge[:, np.newaxis, np.newaxis] * mm_efield_me

            embed.mm_atoms_near.force += force.sum(axis=1)
            self.qm_atoms.force -= force.sum(axis=0)

            self.qm_atoms.qm_energy += energy.sum()

    def corr_scaling(self, embed):
        """Correct forces due to charge scaling."""

        if embed.qmSwitchingType is not None:

            mm_charge = embed.mm_atoms_near.charge
            mm_esp = embed.get_mm_esp()
            scale_deriv = embed.scale_deriv
            dij_min_j = embed.mm_atoms_near.dij_min_j

            energy = mm_charge * mm_esp

            force_corr = -1 * energy[:, np.newaxis] * scale_deriv

            embed.mm_atoms_near.force -= force_corr
            np.add.at(self.qm_atoms.force, dij_min_j, force_corr)

    def corr_virt_mm_atoms(self):
        """Correct forces due to virtual MM charges."""

        if self.n_virt_mm_atoms > 0:

            virt_force = self.mm_atoms.virt_atoms.force.reshape(-1, self._mm1_coeff.size, 3)

            mm1_corr_force = (virt_force * self._mm1_coeff[:, np.newaxis]).sum(axis=1)
            mm2_corr_force = (virt_force * self._mm2_coeff[:, np.newaxis]).sum(axis=1)

            np.add.at(self.mm_atoms.force, self._virt_atom_mm1_idx, mm1_corr_force)
            np.add.at(self.mm_atoms.force, self._virt_atom_mm2_idx, mm2_corr_force)

            self.mm_atoms.virt_atoms.force[:] = 0.
