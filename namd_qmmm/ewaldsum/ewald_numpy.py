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

        # Cancel self energy
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

    def get_recip_esp_near(self, rij, dij=None):
        if dij is None:
            dij2 = np.sum(rij**2, axis=2)
            dij = np.sqrt(dij2)

        recip_esp_near = get_recip_esp_near(dij, self.alpha)

        return recip_esp_near

    def get_recip_efield_near(self, rij, dij=None):

        if dij is None:
            dij2 = np.sum(rij**2, axis=2)
            dij = np.sqrt(dij2)

        recip_efield_near = get_recip_efield_near(rij, dij[:, :, np.newaxis], self.alpha)

        return recip_efield_near


@np.vectorize
def get_recip_esp_near(dij, alpha):
    if dij > 0:
        return math.erf(alpha * dij) / dij
    else:
        return 0.0

@np.vectorize
def get_recip_efield_near(rij, dij, alpha):
    if dij > 0:
        prod = (math.erf(alpha * dij) / dij**3
                - 2 * alpha * math.exp(-1 * alpha**2 * dij**2) / math.sqrt(math.pi) / dij**2)
        recip_efield_near =  prod * rij
    else:
        recip_efield_near = 0.0

    return recip_efield_near
