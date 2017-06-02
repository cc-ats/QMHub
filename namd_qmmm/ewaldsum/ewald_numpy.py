import math
import numpy as np

from .ewald_base import EwaldBase


class EwaldNumPy(EwaldBase):

    def get_recip_esp(self, rij, dij=None, self_energy=False, order='spherical'):
        if order.lower() == 'spherical':
            k = self.recip_lattice_spherical
            fac = self.recip_prefactor_spherical
        elif order.lower() == 'rectangular':
            k = self.recip_lattice
            fac = self.recip_prefactor
        else:
            raise ValueError("Unrecognized order of summation.")

        kr = np.dot(rij, k.T)
        recip_esp = np.sum(fac * np.cos(kr), axis=2)

        if dij is None:
            dij2 = np.sum(rij**2, axis=2)
            dij = np.sqrt(dij2)

        if self_energy:
            recip_esp -= 2 * np.equal(dij, 0) * self.alpha / np.sqrt(np.pi)

        return recip_esp

    def get_recip_efield(self, rij, dij=None, order='spherical'):
        if order.lower() == 'spherical':
            k = self.recip_lattice_spherical
            fac = self.recip_prefactor_spherical
        elif order.lower() == 'rectangular':
            k = self.recip_lattice
            fac = self.recip_prefactor
        else:
            raise ValueError("Unrecognized order of summation.")

        kr = np.dot(rij, k.T)
        recip_efield = np.sum((fac * np.sin(kr))[:, :, :, np.newaxis]
                               * k[np.newaxis, np.newaxis, :, :], axis=2)

        return recip_efield
