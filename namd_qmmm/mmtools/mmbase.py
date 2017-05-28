import copy
import numpy as np

from .. import units

class MMBase(object):

    MMTOOL = None

    def __init__(self, fin=None):

        self.fin = fin
        self.load_system(self.fin)
        self.get_pair_vectors()
        self.get_pair_distances()
        self.sort_qmatoms()

    def get_pair_vectors(self):
        """Get pair-wise vectors between QM and MM atoms."""

        self.rij = (self.qm_position[np.newaxis, :, :]
                    - self.mm_position[:, np.newaxis, :])

        return self.rij

    def get_pair_distances(self):
        """Get pair-wise distances between QM and MM atoms."""

        self.dij2 = np.sum(self.rij**2, axis=2)
        self.dij = np.sqrt(self.dij2)

        return self.dij

    def get_min_distances(self):
        """Get minimum distances between QM and MM atoms."""
        self.dij_min2 = self.dij2[0:self.n_real_mm_atoms, 0:self.n_real_qm_atoms].min(axis=1)
        self.dij_min_j = self.dij2[0:self.n_real_mm_atoms, 0:self.n_real_qm_atoms].argmin(axis=1)
        self.dij_min = np.sqrt(self.dij_min2)

        return self.dij_min

    def sort_qmatoms(self):
        """Sort QM atoms."""
        self.map2sorted = np.concatenate((np.argsort(self.qm_index[0:self.n_real_qm_atoms]),
                                     np.arange(self.n_real_qm_atoms, self.n_qm_atoms)))
        self.map2unsorted = np.argsort(self.map2sorted)

    def absorb_vpntchrgs_mm1(self):
        """Absorb the virtual point charges to MM1."""

        rPntChrgs = self.mm_charge[0:self.n_real_mm_atoms]

        if self.n_virt_mm_atoms > 0:
            vPntChrgs = self.mm_charge[self.n_real_mm_atoms:]
            if self.n_virt_mm_atoms__per_mm2 == 3:
                for i in range(self.n_mm2):
                    rPntChrgs[self.virt_atom_mm1_idx[i]] += vPntChrgs[(i * 3):(i * 3 + 3)].sum()

            elif self.n_virt_mm_atoms_per_mm2 == 2:
                raise NotImplementedError()

        return rPntChrgs

    def split_mm_atoms(self, qmCutoff):
        """Split external point charges into near- and far-fields."""

        nearfield = np.append((self.dij_min <= qmCutoff), np.ones(self.n_virt_mm_atoms, dtype=bool))
        self.mm_charge_near = self.mm_charge[nearfield]
        self.mm_position_near = self.mm_position[nearfield]
        self.rij_near = self.rij[nearfield]
        self.dij_near = self.dij[nearfield]
        self.dij2_near = self.dij2[nearfield]
        self.dij_min_near = self.dij_min[nearfield[0:self.n_real_mm_atoms]]
        self.dij_min2_near = self.dij_min2[nearfield[0:self.n_real_mm_atoms]]
        self.dij_min_j_near = self.dij_min_j[nearfield[0:self.n_real_mm_atoms]]

        self.mm_charge_far = self.absorb_vpntchrgs_mm1()
        self.mm_position_far = self.mm_position[0:self.n_real_mm_atoms]
        self.rij_far = self.rij[0:self.n_real_mm_atoms]
        self.dij_far = self.dij[0:self.n_real_mm_atoms]
        self.dij2_far = self.dij2[0:self.n_real_mm_atoms]

    def scale_charges(self, qmSwitchingType=None, qmCutoff=None, qmSwdist=None):
        """Scale external point charges."""

        if qmSwitchingType is None:
            qmSwdist = 0.0
            self.charge_scale = np.ones(self.n_real_mm_atoms)
            self.scale_deriv = np.zeros(self.n_real_mm_atoms)
        elif qmSwitchingType.lower() == 'shift':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            qmSwdist = 0.0
            self.charge_scale = (1 - self.dij_min2 / qmCutoff2)**2
            self.scale_deriv = 4 * (1 - self.dij_min2 / qmCutoff2) / qmCutoff2
        elif qmSwitchingType.lower() == 'switch':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            if qmSwdist is None:
                qmSwdist = 0.75 * qmCutoff
            if qmCutoff <= qmSwdist:
                raise ValueError("qmCutoff should be greater than qmSwdist.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = qmSwdist**2
            self.charge_scale = ((self.dij_min2 - qmCutoff2)**2
                             * (qmCutoff2 + 2 * self.dij_min2 - 3 * qmSwdist2)
                             / (qmCutoff2 - qmSwdist2)**3
                             * (self.dij_min2 >= qmSwdist2)
                             + (self.dij_min2 < qmSwdist2))
            self.scale_deriv = (12 * (self.dij_min2 - qmSwdist2)
                                   * (qmCutoff2 - self.dij_min2)
                                   / (qmCutoff2 - qmSwdist2)**3)
        elif qmSwitchingType.lower() == 'lrec':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            qmSwdist = 0.0
            scale = 1 - self.dij_min / qmCutoff
            self.charge_scale = 1 - (2 * scale**3 - 3 * scale**2 + 1)**2
            self.scale_deriv = 12 * scale * (2 * scale**3 - 3 * scale**2 + 1) / qmCutoff2
        else:
            raise ValueError("Only 'shift', 'switch', and 'lrec' are supported at the moment.")

        self.scale_deriv *= (self.dij_min > qmSwdist)
        self.scale_deriv = (-1 * self.scale_deriv[:, np.newaxis]
                               * self.rij[range(self.n_real_mm_atoms), self.dij_min_j])

        # Just to be safe
        self.charge_scale *= (self.dij_min < qmCutoff)
        self.scale_deriv *= (self.dij_min < qmCutoff)[:, np.newaxis]

        self.mm_charge_scaled = self.mm_charge * np.append(self.charge_scale, np.ones(self.n_virt_mm_atoms))

    def parse_output(self, qm):
        """Parse the output of QM calculation."""
        if qm.calc_forces:
            self.qm_energy = qm.get_qm_energy() * units.E_AU
            self.qm_force = qm.get_qm_force()[self.map2unsorted] * units.F_AU
            self.qm_charge = qm.get_qm_charge()[self.map2unsorted]
            self.mm_force = qm.get_mm_force() * units.F_AU
            self.mm_esp = qm.get_mm_esp() * units.E_AU
        else:
            self.qm_energy = 0.0
            self.qm_force = np.zeros((self.n_qm_atoms, 3))
            self.mm_force = np.zeros((self.n_real_mm_atoms, 3))

    def corr_elecembed(self):
        """Correct forces due to scaling external point charges in Electrostatic Embedding."""

        fCorr = self.mm_esp[0:self.n_real_mm_atoms] * self.mm_charge[0:self.n_real_mm_atoms]
        fCorr = fCorr[:, np.newaxis] * self.scale_deriv
        self.mm_force[0:self.n_real_mm_atoms] += fCorr

        for i in range(self.n_real_qm_atoms):
            self.qm_force[i] -= fCorr[self.dij_min_j == i].sum(axis=0)

    def corr_mechembed(self):
        """Correct forces and energy due to mechanical embedding."""
        mm_charge_diff = self.mm_charge_mm[0:self.n_real_mm_atoms] - self.mm_charge_qm[0:self.n_real_mm_atoms]

        fCorr = (-1 * units.KE * mm_charge_diff[:, np.newaxis] * self.qm_charge_me[np.newaxis, :]
                 / self.dij[0:self.n_real_mm_atoms]**3)
        fCorr = fCorr[:, :, np.newaxis] * self.rij[0:self.n_real_mm_atoms]

        if self.n_virt_mm_atoms > 0:
            for i in range(self.n_virt_qm_atoms):
                fCorr[self.mm2_local_idx[i], self.qm_host_local_idx[i]] = 0.0

        self.mm_force[0:self.n_real_mm_atoms] += fCorr.sum(axis=1)
        self.qm_force -= fCorr.sum(axis=0)

        if hasattr(self, 'mm_charge_scaled'):
            fCorr = (units.KE * self.mm_charge[0:self.n_real_mm_atoms, np.newaxis]
                     * self.qm_charge_me[np.newaxis, :]
                     / self.dij[0:self.n_real_mm_atoms])
            if self.n_virt_mm_atoms > 0:
                for i in range(self.n_virt_qm_atoms):
                    fCorr[self.mm2_local_idx[i], self.qm_host_local_idx[i]] = 0.0
            fCorr = np.sum(fCorr, axis=1)
            fCorr = fCorr[:, np.newaxis] * self.scale_deriv

            if self.mm_charge_mm is self.mm_charge_scaled:
                fCorr *= -1

            self.mm_force[0:self.n_real_mm_atoms] -= fCorr
            for i in range(self.n_real_qm_atoms):
                self.qm_force[i] += fCorr[self.dij_min_j == i].sum(axis=0)

        eCorr = (units.KE * mm_charge_diff[:, np.newaxis] * self.qm_charge_me[np.newaxis, :]
                 / self.dij[0:self.n_real_mm_atoms])

        if self.n_virt_mm_atoms > 0:
            for i in range(self.n_virt_qm_atoms):
                eCorr[self.mm2_local_idx[i], self.qm_host_local_idx[i]] = 0.0

        self.qm_energy += eCorr.sum()

    def corr_vpntchrgs(self):
        """Correct forces due to virtual external point charges."""
        if self.n_virt_mm_atoms > 0:
            if self.n_virt_mm_atoms_per_mm2 == 3:
                for i in range(self.n_mm2):
                    self.mm_force[self.virt_atom_mm2_idx[i]] += self.mm_force[self.n_real_mm_atoms + i*3]

                    self.mm_force[self.virt_atom_mm2_idx[i]] += self.mm_force[self.n_real_mm_atoms + i*3 + 1] * 0.94
                    self.mm_force[self.virt_atom_mm1_idx[i]] += self.mm_force[self.n_real_mm_atoms + i*3 + 1] * 0.06

                    self.mm_force[self.virt_atom_mm2_idx[i]] += self.mm_force[self.n_real_mm_atoms + i*3 + 2] * 1.06
                    self.mm_force[self.virt_atom_mm1_idx[i]] += self.mm_force[self.n_real_mm_atoms + i*3 + 2] * -0.06

                self.mm_force[self.n_real_mm_atoms:] = 0.

            elif self.n_virt_mm_atoms_per_mm2 == 2:
                raise NotImplementedError()
        else:
            pass
