import copy
import numpy as np


class MMBase(object):

    MMTOOL = None

    HARTREE2KCALMOL = 6.275094737775374e+02
    BOHR2ANGSTROM = 5.2917721067e-01
    KE = HARTREE2KCALMOL * BOHR2ANGSTROM

    def __init__(self, fin=None):

        self.fin = fin
        self.load_system(self.fin)
        self.get_pair_vectors()
        self.get_pair_distances()
        self.sort_qmatoms()

    def get_pair_vectors(self):
        """Get pair-wise vectors between QM and MM atoms."""

        self.rij = (self.qmPos[np.newaxis, :, :]
                    - self.pntPos[:, np.newaxis, :])

        return self.rij

    def get_pair_distances(self):
        """Get pair-wise distances between QM and MM atoms."""

        self.dij2 = np.sum(self.rij**2, axis=2)
        self.dij = np.sqrt(self.dij2)

        return self.dij

    def get_min_distances(self):
        """Get minimum distances between QM and MM atoms."""
        self.dij_min2 = self.dij2[0:self.numRPntChrgs, 0:self.numRealQMAtoms].min(axis=1)
        self.dij_min_j = self.dij2[0:self.numRPntChrgs, 0:self.numRealQMAtoms].argmin(axis=1)
        self.dij_min = np.sqrt(self.dij_min2)

        return self.dij_min

    def get_pntchrg_types(self, qmCutoff=None):
        """Get the types of external point charges."""

        self.pntChrgTypes = np.zeros(self.numPntChrgs, dtype=int) 
        self.pntChrgTypes[np.where(self.dij_min > qmCutoff)] = 1
        self.pntChrgTypes[self.numRPntChrgs:] = -1

        return self.pntChrgTypes

    def sort_qmatoms(self):
        """Sort QM atoms."""
        self.map2sorted = np.concatenate((np.argsort(self.qmIdx[0:self.numRealQMAtoms]),
                                     np.arange(self.numRealQMAtoms, self.numQMAtoms)))
        self.map2unsorted = np.argsort(self.map2sorted)

    def scale_charges(self, qmSwitchingType=None, qmCutoff=None, qmSwdist=None):
        """Scale external point charges."""

        if qmSwitchingType.lower() == 'shift':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = 0.0
            self.pntScale = (1 - self.dij_min2 / qmCutoff2)**2
            self.pntScale_deriv = 4 * (1 - self.dij_min2 / qmCutoff2) / qmCutoff2
        elif qmSwitchingType.lower() == 'switch':
            if qmCutoff is None or qmSwdist is None:
                raise ValueError("We need qmCutoff and qmSwdist here.")
            if qmCutoff <= qmSwdist:
                raise ValueError("qmCutoff should be greater than qmSwdist.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = qmSwdist**2
            self.pntScale = ((self.dij_min2 - qmCutoff2)**2
                             * (qmCutoff2 + 2 * self.dij_min2 - 3 * qmSwdist2)
                             / (qmCutoff2 - qmSwdist2)**3
                             * (self.dij_min2 >= qmSwdist2)
                             + (self.dij_min2 < qmSwdist2))
            self.pntScale_deriv = (12 * (self.dij_min2 - qmSwdist2)
                                   * (qmCutoff2 - self.dij_min2)
                                   / (qmCutoff2 - qmSwdist2)**3)
        elif qmSwitchingType.lower() == 'lrec':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = 0.0
            scale = 1 - self.dij_min / qmCutoff
            self.pntScale = 1 - (2 * scale**3 - 3 * scale**2 + 1)**2
            self.pntScale_deriv = 12 * scale * (2 * scale**3 - 3 * scale**2 + 1) / qmCutoff2
        else:
            raise ValueError("Only 'shift', 'switch', and 'lrec' are supported at the moment.")

        self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                * (self.pntPos[0:self.numRPntChrgs]
                                - self.qmPos[self.dij_min_j]))
        self.pntScale_deriv *= (self.dij_min2 > qmSwdist2)[:, np.newaxis]

        # Just to be safe
        self.pntScale *= (self.dij_min < qmCutoff)
        self.pntScale_deriv *= (self.dij_min < qmCutoff)[:, np.newaxis]

        self.pntScale = np.append(self.pntScale, np.ones(self.numVPntChrgs))
        self.pntChrgsScld = self.pntChrgs * self.pntScale

    def parse_output(self, qm):
        """Parse the output of QM calculation."""
        self.qmEnergy = qm.get_qmenergy() * self.HARTREE2KCALMOL
        if qm.calc_forces:
            self.qmForces = qm.get_qmforces()[self.map2unsorted] * self.HARTREE2KCALMOL / self.BOHR2ANGSTROM
            self.pntChrgForces = qm.get_pntchrgforces() * self.HARTREE2KCALMOL / self.BOHR2ANGSTROM
            self.qmChrgs = qm.get_qmchrgs()[self.map2unsorted]
            self.pntESP = qm.get_pntesp() * self.HARTREE2KCALMOL
        else:
            self.qmForces = np.zeros((self.numQMAtoms, 3))
            self.pntChrgForces = np.zeros((self.numRPntChrgs, 3))

    def corr_elecembed(self):
        """Correct forces due to scaling external point charges in Electrostatic Embedding."""

        fCorr = self.pntESP[0:self.numRPntChrgs] * self.pntChrgs[0:self.numRPntChrgs]
        fCorr = fCorr[:, np.newaxis] * self.pntScale_deriv
        self.pntChrgForces[0:self.numRPntChrgs] += fCorr

        for i in range(self.numRealQMAtoms):
            self.qmForces[i] -= fCorr[self.dij_min_j == i].sum(axis=0)

    def corr_mechembed(self):
        """Correct forces and energy due to mechanical embedding."""
        pntChrgsD = self.pntChrgs4MM[0:self.numRPntChrgs] - self.pntChrgs4QM[0:self.numRPntChrgs]

        fCorr = (-1 * self.KE * pntChrgsD[:, np.newaxis] * self.qmChrgs4MM[np.newaxis, :]
                 / self.dij[0:self.numRPntChrgs]**3)
        fCorr = fCorr[:, :, np.newaxis] * self.rij[0:self.numRPntChrgs]

        if self.numVPntChrgs > 0:
            for i in range(self.numMM1):
                fCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0

        self.pntChrgForces[0:self.numRPntChrgs] += fCorr.sum(axis=1)
        self.qmForces -= fCorr.sum(axis=0)

        if hasattr(self, 'pntChrgsScld'):
            fCorr = (self.KE * self.pntChrgs[0:self.numRPntChrgs, np.newaxis]
                     * self.qmChrgs4MM[np.newaxis, :]
                     / self.dij[0:self.numRPntChrgs])
            if self.numVPntChrgs > 0:
                for i in range(self.numMM1):
                    fCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0
            fCorr = np.sum(fCorr, axis=1)
            fCorr = fCorr[:, np.newaxis] * self.pntScale_deriv

            if self.pntChrgs4MM is self.pntChrgsScld:
                fCorr *= -1

            self.pntChrgForces[0:self.numRPntChrgs] -= fCorr
            for i in range(self.numRealQMAtoms):
                self.qmForces[i] += fCorr[self.dij_min_j == i].sum(axis=0)

        eCorr = (self.KE * pntChrgsD[:, np.newaxis] * self.qmChrgs4MM[np.newaxis, :]
                 / self.dij[0:self.numRPntChrgs])

        if self.numVPntChrgs > 0:
            for i in range(self.numMM1):
                eCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0

        self.qmEnergy += eCorr.sum()

    def corr_vpntchrgs(self):
        """Correct forces due to virtual external point charges."""
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                for i in range(self.numMM2):
                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3]

                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.94
                    self.pntChrgForces[self.mm1VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.06

                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * 1.06
                    self.pntChrgForces[self.mm1VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * -0.06

                self.pntChrgForces[self.numRPntChrgs:] = 0.

            elif self.numVPntChrgsPerMM2 == 2:
                raise NotImplementedError()
        else:
            pass
