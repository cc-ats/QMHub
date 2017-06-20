import math
import numpy as np


class EwaldBase(object):

    def __init__(self, cell_basis, cutoff, tolerance=1e-4, alpha=None, kmax=None):

        self.cell_basis = cell_basis
        self.cutoff = cutoff
        self.tolerance = tolerance

        self._alpha = alpha
        self._kmax = kmax

        self._volume = None
        self._recip_basis = None

        self._recip_grids = None
        self._recip_lattice = None
        self._recip_prefactor = None

        self._recip_grids_spherical = None
        self._recip_lattice_spherical = None
        self._recip_prefactor_spherical = None

    @property
    def volume(self):
        if self._volume is None:
            self._volume = np.linalg.det(self.cell_basis)
        return self._volume

    @property
    def recip_basis(self):
        if self._recip_basis is None:
            self._recip_basis = 2 * np.pi * np.linalg.inv(self.cell_basis).T
        return self._recip_basis

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = math.sqrt(-1 * math.log(self.tolerance)) / self.cutoff
        return self._alpha

    @property
    def kmax(self):
        if self._kmax is None:
            self._kmax = np.ceil(2 * self.alpha**2 * self.cutoff / np.diag(self.recip_basis)).astype(int)
        return self._kmax

    @property
    def recip_grids(self):
        if self._recip_grids is None:
            kmax = self.kmax
            grids = np.mgrid[-kmax[0]:kmax[0]+1, -kmax[1]:kmax[1]+1, -kmax[2]:kmax[2]+1]
            grids = np.rollaxis(grids, 0, 4)
            grids = grids.reshape((kmax[0]*2 + 1) * (kmax[1]*2 + 1) * (kmax[2]*2 + 1), 3)

            # Delete the central unit
            grids = grids[~np.all(grids == 0, axis=1)]

            self._recip_grids = grids

        return self._recip_grids

    @property
    def recip_grids_spherical(self):
        if self._recip_grids_spherical is None:
            self._recip_grids_spherical = self.recip_grids[np.sum((self.recip_grids * np.diag(self.recip_basis))**2, axis=1) <= np.max(self.kmax * np.diag(self.recip_basis))**2]
        return self._recip_grids_spherical

    @property
    def recip_lattice(self):
        if self._recip_lattice is None:
            self._recip_lattice = np.dot(self.recip_grids, self.recip_basis)
        return self._recip_lattice

    @property
    def recip_lattice_spherical(self):
        if self._recip_lattice_spherical is None:
            self._recip_lattice_spherical = np.dot(self.recip_grids_spherical, self.recip_basis)
        return self._recip_lattice_spherical

    @property
    def recip_prefactor(self):
        if self._recip_prefactor is None:
            k = self.recip_lattice
            k2 = np.sum(k * k, axis=1)
            self._recip_prefactor = (4 * np.pi / self.volume) * np.exp(-1 * k2 / (4 * self.alpha**2)) / k2
        return self._recip_prefactor

    @property
    def recip_prefactor_spherical(self):
        if self._recip_prefactor_spherical is None:
            k = self.recip_lattice_spherical
            k2 = np.sum(k * k, axis=1)
            self._recip_prefactor_spherical = (4 * np.pi / self.volume) * np.exp(-1 * k2 / (4 * self.alpha**2)) / k2
        return self._recip_prefactor_spherical
