import math
import numpy as np
import numba as nb

from .ewald_base import EwaldBase


class EwaldNumba(EwaldBase):

    def get_recip_esp(self, rij, dij=None, self_energy=False, order='spherical'):
        if order.lower() == 'spherical':
            k = self.recip_lattice_spherical
            fac = self.recip_prefactor_spherical
        elif order.lower() == 'rectangular':
            k = self.recip_lattice
            fac = self.recip_prefactor
        else:
            raise ValueError("Unrecognized order of summation.")

        recip_esp = np.zeros((rij.shape[0], rij.shape[1]))
        get_recip_esp(rij, k, fac, recip_esp)

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

        recip_efield = np.zeros((rij.shape[0], rij.shape[1], 3))
        get_recip_efield(rij, k, fac, recip_efield)

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

        recip_efield_near = np.zeros((rij.shape[0], rij.shape[1], 3))
        get_recip_efield_near(rij, dij, self.alpha, recip_efield_near)

        return recip_efield_near


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[0])],
                '(m),(n,m),(n)->()', nopython=True, target='parallel', cache=True)
def get_recip_esp(rij, k, fac, recip_esp):
    kr = np.dot(k, rij)
    for n in range(len(fac)):
        recip_esp[0] += fac[n] * math.cos(kr[n])


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:])],
                '(m),(n,m),(n)->(m)', nopython=True, target='parallel', cache=True)
def get_recip_efield(rij, k, fac, recip_efield):
    kr = np.dot(k, rij)
    for n in range(len(fac)):
        prod = fac[n] * math.sin(kr[n])
        recip_efield[0] += prod * k[n, 0]
        recip_efield[1] += prod * k[n, 1]
        recip_efield[2] += prod * k[n, 2]


@nb.vectorize([nb.float64(nb.float64, nb.float64)],
              nopython=True, target='parallel', cache=True)
def get_recip_esp_near(dij, alpha):
    if dij > 0:
        return math.erf(alpha * dij) / dij
    else:
        return 0.0


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:])],
                '(m),(),()->(m)', nopython=True, target='parallel', cache=True)
def get_recip_efield_near(rij, dij, alpha, recip_efield_near):
    if dij[0] > 0:
        prod = (math.erf(alpha[0] * dij[0]) / dij[0]**3
                - 2 * alpha[0] * math.exp(-1 * alpha[0]**2 * dij[0]**2) / math.sqrt(math.pi) / dij[0]**2)
        recip_efield_near[0] =  prod * rij[0]
        recip_efield_near[1] =  prod * rij[1]
        recip_efield_near[2] =  prod * rij[2]
    else:
        recip_efield_near[0] = 0.0
        recip_efield_near[1] = 0.0
        recip_efield_near[2] = 0.0
