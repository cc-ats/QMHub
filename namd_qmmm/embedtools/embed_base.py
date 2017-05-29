import warnings
import numpy as np

from .. import units


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

        # Check if MM charges need to be switched
        self.check_qm_switching_type()

        # Pass system infomation
        self.qm_atoms = system.qm_atoms
        self.mm_atoms = system.mm_atoms
        self.cell_basis = system.cell_basis
        self.cell_origin = system.cell_origin

        # Initialize properties
        self._deriv_r = None
        self._coulomb_potential = None
        self._coulomb_field = None

        self._qm_esp_near = None
        self._qm_efield_near = None

        # Get QM charges for Mechanical Embedding
        self.get_qm_charge_me()
        self.qm_atoms.charge_me = self.qm_charge_me

        # Split MM atoms
        self.split_mm_atoms()

        # Scale MM charges in the near field
        self.scale_mm_charges()

        # Get MM charges
        self.get_mm_charge()

    @classmethod
    def check_embed(cls, embed_near, embed_far):
        return (embed_near.lower() == cls.EMBEDNEAR.lower() and
                embed_far.lower() == cls.EMBEDFAR.lower())

    @staticmethod
    def check_unitcell(system):
        pass

    def check_qm_switching_type(self):
        if self.qmSwitchingType is None:
            warnings.warn("Not switching MM charges might cause discontinuity at the cutoff boundary.")

    def get_qm_charge_me(self):
        self.qm_charge_me = self.qmRefCharge

    def get_near_mask(self):
        return np.ones(self.mm_atoms.n_atoms, dtype=bool)

    def split_mm_atoms(self):
        """Get MM atoms in the near field."""

        near_mask = self.get_near_mask()
        self.mm_atoms_near = self.mm_atoms.mask_atoms(near_mask)
        self.mm_atoms_far = self.mm_atoms.real_atoms

    def scale_mm_charges(self):
        """Scale external point charges."""

        if self.qmSwitchingType is None:
            self.charge_scale = np.ones(self.mm_atoms_near.n_atoms, dtype=float)
            self.scale_deriv = np.zeros(self.mm_atoms_near.n_atoms, dtype=float)
            self.mm_atoms_near.charge_near = self.mm_atoms_near.charge
            self.mm_atoms_near.charge_comp = None
        else:
            if self.qmCutoff is None:
                raise ValueError("We need qmCutoff here.")

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
                charge_scale = 1 - (2 * scale**3 - 3 * scale**3 + 1)**2
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

    def get_mm_charge(self):
        """Get MM atom charges."""

        self.mm_atoms_near.charge_me = None
        self.mm_atoms_near.charge_eeq = None
        self.mm_atoms_near.charge_eed = None
        self.mm_atoms_far.charge_eeq = None

    def get_mm_esp_me(self):

        coulomb_mask = self.mm_atoms_near.coulomb_mask

        return self.coulomb_potential * self.qm_charge_me * coulomb_mask

    def get_mm_esp_eeq(self):

        return self.coulomb_potential * self.qm_atoms.qm_charge

    def get_mm_esp_eed(self):

        return self.mm_atoms_near.esp_eed

    @property
    def deriv_r(self):
        if self._deriv_r is None:
            rij = self.mm_atoms_near.rij
            dij2 = self.mm_atoms_near.dij2
            self._deriv_r = rij / dij2[:, :, np.newaxis]
        return self._deriv_r

    @property
    def coulomb_potential(self):
        if self._coulomb_potential is None:
            self._coulomb_potential = units.KE / self.mm_atoms_near.dij
        return self._coulomb_potential

    @property
    def coulomb_field(self):
        if self._coulomb_field is None:
            self._coulomb_field = self.coulomb_potential[:, :, np.newaxis] * self.deriv_r
        return self._coulomb_field

    @property
    def qm_esp_near(self):
        if self._qm_esp_near is None:
            self._qm_esp_near = np.sum(self.coulomb_potential * self.mm_atoms_near.charge_eeq[:, np.newaxis], axis=0)
        return self._qm_esp_near

    @property
    def qm_efield_near(self):
        if self._qm_efield_near is None:
            self._qm_efield_near = self.coulomb_field * self.mm_atoms_near.charge_eeq[:, np.newaxis, np.newaxis]
        return self._qm_efield_near
