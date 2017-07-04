import warnings
import numpy as np

from .. import units
from .. import ewaldsum


class EmbedBase(object):

    def __init__(self, system, qmRefCharge, qmSwitchingType, qmCutoff, qmSwdist):
        """
        Creat a EmbedBase object.
        """

        self.qmRefCharge = qmRefCharge
        self.qmSwitchingType = qmSwitchingType
        self.qmCutoff = qmCutoff
        self.qmSwdist = qmSwdist

        # Check if unit cell is complete
        self.check_unitcell(system)

        # Pass system infomation
        self.qm_atoms = system.qm_atoms
        self.mm_atoms = system.mm_atoms
        self.cell_basis = system.cell_basis
        self.cell_origin = system.cell_origin

        self.ewald_cutoff = np.diag(self.cell_basis).min() / 2.0
        self.ewaldsum = ewaldsum.EwaldSum(self.cell_basis, self.ewald_cutoff)

        # Initialize properties
        self._coulomb_deriv = None
        self._coulomb_potential = None
        self._coulomb_field = None
        self._ewald_potential_qmmm = None
        self._ewald_potential_qmqm = None
        self._ewald_field_qmmm = None
        self._ewald_field_qmqm = None

        self._qmmm_esp_near = None
        self._qmmm_esp_far = None
        self._qmqm_esp_far = None

        self._qmmm_efield_near = None
        self._qmmm_efield_far = None
        self._qmqm_efield_far = None

        # Get QM charges for Mechanical Embedding
        self.get_qm_charge_me()
        self.qm_atoms.charge_me = self.qm_charge_me

        self.get_qm_charge_eeq()

        # Split MM atoms
        self.split_mm_atoms()

        # Scale MM charges in the near field
        self.scale_mm_charges()

        # Get MM charges
        self.get_mm_charge()

    @staticmethod
    def check_unitcell(system):
        pass

    def get_qm_charge_me(self):
        self.qm_charge_me = self.qm_atoms.charge

    def get_qm_charge_eeq(self):
        self.qm_charge_eeq = None

    def get_near_mask(self):
        return np.ones(self.mm_atoms.n_atoms, dtype=bool)

    def split_mm_atoms(self):
        """Get MM atoms in the near field."""

        near_mask = self.get_near_mask()
        self.mm_atoms_near = self.mm_atoms.mask_atoms(near_mask)
        self.mm_atoms_far = self.mm_atoms.real_atoms

    def scale_mm_charges(self):
        """Scale external point charges."""

        if self.qmCutoff is None:
            raise ValueError("We need qmCutoff here.")

        if self.qmSwitchingType is None:
            warnings.warn("Not switching MM charges might cause discontinuity at the cutoff boundary.")

            cutoff = self.qmCutoff
            dij_min = self.mm_atoms_near.dij_min

            charge_scale = np.ones(self.mm_atoms_near.n_atoms, dtype=float)
            scale_deriv = np.zeros((self.mm_atoms_near.n_atoms, 3), dtype=float)

            charge_scale *= (dij_min < cutoff)
            scale_deriv *= (dij_min < cutoff)[:, np.newaxis]

        else:
            cutoff = self.qmCutoff
            cutoff2 = cutoff**2
            swdist = self.qmSwdist

            rij = self.mm_atoms_near.rij
            dij_min = self.mm_atoms_near.dij_min
            dij_min2 = self.mm_atoms_near.dij_min2
            dij_min_j = self.mm_atoms_near.dij_min_j

            if self.qmSwitchingType.lower() == 'shift':
                swdist = 0.0
                charge_scale = (1 - dij_min2 / cutoff2)**2
                scale_deriv = 4 * (1 - dij_min2 / cutoff2) / cutoff2
            elif self.qmSwitchingType.lower() == 'switch':
                if swdist is None:
                    swdist = 0.75 * cutoff
                if cutoff <= swdist:
                    raise ValueError("qmCutoff should be greater than qmSwdist.")
                swdist2 = swdist**2
                charge_scale = ((dij_min2 - cutoff2)**2
                                * (cutoff2 + 2 * dij_min2 - 3 * swdist2)
                                / (cutoff2 - swdist2)**3
                                * (dij_min2 >= swdist2)
                                + (dij_min2 < swdist2))
                scale_deriv = (12 * (dij_min2 - swdist2)
                               * (cutoff2 - dij_min2)
                               / (cutoff2 - swdist2)**3)
            elif self.qmSwitchingType.lower() == 'lrec':
                swdist = 0.0
                scale = 1 - dij_min / cutoff
                charge_scale = 1 - (2 * scale**3 - 3 * scale**2 + 1)**2
                scale_deriv = 12 * scale * (2 * scale**3 - 3 * scale**2 + 1) / cutoff2
            else:
                raise ValueError("Only 'shift', 'switch', and 'lrec' are supported at the moment.")

            scale_deriv *= (dij_min > swdist)
            scale_deriv = (-1 * scale_deriv[:, np.newaxis]
                           * rij[range(len(dij_min)), dij_min_j])

            # Just to be safe
            charge_scale *= (dij_min < cutoff)
            scale_deriv *= (dij_min < cutoff)[:, np.newaxis]

        self.charge_scale = charge_scale
        self.scale_deriv = scale_deriv

        self.mm_atoms_near.charge_near = self.mm_atoms_near.charge * self.charge_scale
        self.mm_atoms_near.charge_comp = self.mm_atoms_near.charge * (1 - self.charge_scale)

        if np.all(self.mm_atoms_near.charge_comp == 0.0):
            self.mm_atoms_near.charge_comp = None

    def get_mm_charge(self):
        """Get MM atom charges."""

        self.mm_atoms_near.charge_me = None
        self.mm_atoms_near.charge_eeq = None
        self.mm_atoms_near.charge_eed = None
        self.mm_atoms_far.charge_eeq = None
        self.mm_atoms_far.charge_eed = None

    def get_mm_esp_me(self):

        coulomb_mask = self.mm_atoms_near.coulomb_mask

        return self.coulomb_potential * self.qm_charge_me * coulomb_mask

    def get_mm_esp_eeq(self):

        return self.coulomb_potential * self.qm_atoms.qm_charge

    def get_mm_esp_eed(self):

        return self.mm_atoms_near.esp_eed

    @property
    def coulomb_deriv(self):
        if self._coulomb_deriv is None:
            rij = self.mm_atoms_near.rij
            dij2 = self.mm_atoms_near.dij2
            self._coulomb_deriv = rij / dij2[:, :, np.newaxis]
        return self._coulomb_deriv

    @property
    def coulomb_potential(self):
        if self._coulomb_potential is None:
            self._coulomb_potential = units.KE / self.mm_atoms_near.dij
        return self._coulomb_potential

    @property
    def coulomb_field(self):
        if self._coulomb_field is None:
            self._coulomb_field = self.coulomb_potential[:, :, np.newaxis] * self.coulomb_deriv
        return self._coulomb_field

    @property
    def ewald_potential_qmmm(self):
        if self._ewald_potential_qmmm is None:
            self._ewald_potential_qmmm = units.KE * self.ewaldsum.get_recip_esp(
                rij=self.mm_atoms_far.rij, dij=self.mm_atoms_far.dij)

            near_real_mask = self.mm_atoms_near._atom_mask[self.mm_atoms_near.real_atoms._indices]

            self._ewald_potential_qmmm[near_real_mask] -= units.KE * self.ewaldsum.get_recip_esp_near(
                rij=self.mm_atoms_near.real_atoms.rij, dij=self.mm_atoms_near.real_atoms.dij)

        return self._ewald_potential_qmmm

    @property
    def ewald_potential_qmqm(self):
        if self._ewald_potential_qmqm is None:
            self._ewald_potential_qmqm = units.KE * (self.ewaldsum.get_recip_esp(
                rij=self.qm_atoms.rij, dij=self.qm_atoms.dij, self_energy=True) -
                self.ewaldsum.get_recip_esp_near(rij=self.qm_atoms.rij, dij=self.qm_atoms.dij))
        return self._ewald_potential_qmqm

    @property
    def ewald_field_qmmm(self):
        if self._ewald_field_qmmm is None:
            self._ewald_field_qmmm = units.KE * self.ewaldsum.get_recip_efield(
                rij=self.mm_atoms_far.rij, dij=self.mm_atoms_far.dij)

            near_real_mask = self.mm_atoms_near._atom_mask[self.mm_atoms.real_atoms._indices]

            self._ewald_field_qmmm[near_real_mask] -= units.KE * self.ewaldsum.get_recip_efield_near(
                rij=self.mm_atoms_near.real_atoms.rij, dij=self.mm_atoms_near.real_atoms.dij)

        return self._ewald_field_qmmm

    @property
    def ewald_field_qmqm(self):
        if self._ewald_field_qmqm is None:
            self._ewald_field_qmqm = units.KE * (self.ewaldsum.get_recip_efield(
                rij=self.qm_atoms.rij, dij=self.qm_atoms.dij) -
                self.ewaldsum.get_recip_efield_near(rij=self.qm_atoms.rij, dij=self.qm_atoms.dij))
        return self._ewald_field_qmqm

    @property
    def qmmm_esp_near(self):
        if self._qmmm_esp_near is None:
            if self.mm_atoms_near.charge_eeq is not None:
                self._qmmm_esp_near = self.coulomb_potential * self.mm_atoms_near.charge_eeq[:, np.newaxis]
        return self._qmmm_esp_near

    @property
    def qmmm_esp_far(self):
        if self._qmmm_esp_far is None:
            if self.mm_atoms_far.charge_eeq is not None:
                self._qmmm_esp_far = self.ewald_potential_qmmm * self.mm_atoms_far.charge_eeq[:, np.newaxis]
        return self._qmmm_esp_far

    @property
    def qmqm_esp_far(self):
        if self._qmqm_esp_far is None:
            if self.qm_charge_eeq is not None:
                self._qmqm_esp_far = self.ewald_potential_qmqm * self.qm_charge_eeq[:, np.newaxis]
        return self._qmqm_esp_far

    @property
    def qmmm_efield_near(self):
        if self._qmmm_efield_near is None:
            if self.mm_atoms_near.charge_eeq is not None:
                self._qmmm_efield_near = self.coulomb_field * self.mm_atoms_near.charge_eeq[:, np.newaxis, np.newaxis]
        return self._qmmm_efield_near

    @property
    def qmmm_efield_far(self):
        if self._qmmm_efield_far is None:
            if self.mm_atoms_far.charge_eeq is not None:
                self._qmmm_efield_far = self.ewald_field_qmmm * self.mm_atoms_far.charge_eeq[:, np.newaxis, np.newaxis]
        return self._qmmm_efield_far

    @property
    def qmqm_efield_far(self):
        if self._qmqm_efield_far is None:
            if self.qm_charge_eeq is not None:
                self._qmqm_efield_far = self.ewald_field_qmqm * self.qm_charge_eeq[:, np.newaxis, np.newaxis]
        return self._qmqm_efield_far
