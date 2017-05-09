import os
import subprocess as sp
import numpy as np


class QMBase(object):

    QMTOOL = None

    HARTREE2KCALMOL = 6.275094737775374e+02
    HARTREE2EV = 2.721138602e+01
    BOHR2ANGSTROM = 5.2917721067e-01
    KE = HARTREE2KCALMOL * BOHR2ANGSTROM

    def __init__(self, fin, charge=None, mult=None, pbc=None):
        """
        Creat a QM object.
        """
        if fin is not None:
            self.fin = os.path.abspath(fin)
        else:
            raise ValueError("We need the QMMM file passed from NAMD.")
        self.baseDir = os.path.dirname(self.fin) + "/"
        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1
        if pbc is not None:
            self.pbc = pbc
        else:
            raise ValueError("Please set 'pbc' for QM calculation.")

        # Load system information
        sysList = np.genfromtxt(fin, dtype=int, max_rows=1, unpack=True)
        # Number of QM atoms including linking atoms
        self.numQMAtoms = sysList[0]
        # Number of external external point charges including virtual particles
        self.numPntChrgs = sysList[1]
        # Number of total atoms in the whole system
        self.numAtoms = sysList[2]
        # Number of current step
        self.stepNum = sysList[3]
        # Number of total steps to run
        self.numSteps = sysList[4]

        # Load QM information
        qmList = np.genfromtxt(fin, dtype=None, skip_header=1,
                               max_rows=self.numQMAtoms)
        # Positions of QM atoms
        self.qmPos = np.column_stack((qmList['f0'],
                                      qmList['f1'],
                                      qmList['f2']))
        # Elements of QM atoms
        self.qmElmnts = np.char.capitalize(np.core.defchararray.decode(qmList['f3']))
        # Charges of QM atoms
        self.qmChrgs0 = qmList['f4']
        # Indexes of QM atoms
        self.qmIdx = qmList['f5']

        # Number of MM1 atoms which equals to number of linking atoms
        self.numMM1 = np.count_nonzero(self.qmIdx == -1)
        # Number of Real QM atoms
        self.numRealQMAtoms = self.numQMAtoms - self.numMM1

        # Load external point charge information
        pntList = np.genfromtxt(fin, dtype=None, skip_header=1+self.numQMAtoms,
                                max_rows=self.numPntChrgs)
        # Positions of external point charges
        self.pntPos = np.column_stack((pntList['f0'],
                                       pntList['f1'],
                                       pntList['f2']))
        # Charges of external point charges
        self.pntChrgs = pntList['f3']
        # Indexes of external point charges
        self.pntIdx = pntList['f4']
        # Indexes of QM atoms MM1 atoms bonded to
        self.pntBondedToIdx = pntList['f5']

        # Local indexes of MM1 and QM host atoms
        if self.numMM1 > 0:
            self.mm1LocalIdx = np.where(self.pntBondedToIdx != -1)[0]
            self.qmHostLocalIdx = self.pntBondedToIdx[self.mm1LocalIdx]
        # Number of virtual external point charges
        self.numVPntChrgs = np.count_nonzero(self.pntIdx == -1)
        # Number of real external point charges
        self.numRPntChrgs = self.numPntChrgs - self.numVPntChrgs

        # Numbers of MM2 atoms and virtual external point charges per MM2 atom
        if self.numVPntChrgs > 0:
            if self.pntChrgs[-1] + self.pntChrgs[-2] < 0.00001:
                self.numVPntChrgsPerMM2 = 3
            elif self.pntChrgs[-1] + self.pntChrgs[-2] * 2 < 0.00001:
                self.numVPntChrgsPerMM2 = 2
            else:
                raise ValueError('Something is wrong with point charge alterations.')

            self.numMM2 = self.numVPntChrgs // self.numVPntChrgsPerMM2

        # Local indexes of MM1 and MM2 atoms the virtual point charges belong to
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                mm1VPos = np.zeros((self.numMM2, 3), dtype=float)
                mm2VPos = np.zeros((self.numMM2, 3), dtype=float)
                for i in range(self.numMM2):
                    mm1VPos[i] = (self.pntPos[self.numRPntChrgs + i*3 + 1]
                                  - self.pntPos[self.numRPntChrgs + i*3]
                                  * 0.94) / 0.06
                    mm2VPos[i] = self.pntPos[self.numRPntChrgs + i*3]

                self.mm1VIdx = np.zeros(self.numMM2, dtype=int)
                self.mm2VIdx = np.zeros(self.numMM2, dtype=int)
                for i in range(self.numMM2):
                    for j in range(self.numMM1):
                        if np.abs(mm1VPos[i] - self.pntPos[self.mm1LocalIdx[j]]).sum() < 0.001:
                            self.mm1VIdx[i] = self.mm1LocalIdx[j]
                            break
                for i in range(self.numMM2):
                    for j in range(self.numRPntChrgs):
                        if np.abs(mm2VPos[i] - self.pntPos[j]).sum() < 0.001:
                            self.mm2VIdx[i] = j
                            break
                self.mm2LocalIdx = []
                for i in range(self.numMM1):
                    self.mm2LocalIdx.append(self.mm2VIdx[self.mm1VIdx == self.mm1LocalIdx[i]])
            elif self.numVPntChrgsPerMM2 == 2:
                raise NotImplementedError()

        # Sort QM atoms
        self.map2sorted = np.concatenate((np.argsort(self.qmIdx[0:self.numRealQMAtoms]),
                                     np.arange(self.numRealQMAtoms, self.numQMAtoms)))
        self.map2unsorted = np.argsort(self.map2sorted)

        self.qmElmntsSorted = self.qmElmnts[self.map2sorted]
        self.qmPosSorted = self.qmPos[self.map2sorted]
        self.qmIdxSorted = self.qmIdx[self.map2sorted]

        # Pair-wise vectors between QM and MM atoms
        self.rij = (self.qmPos[np.newaxis, :, :]
                    - self.pntPos[:, np.newaxis, :])
        # Pair-wise distances between QM and MM atoms
        self.dij2 = np.sum(self.rij**2, axis=2)
        self.dij = np.sqrt(self.dij2)

        # Load unit cell information
        if self.pbc:
            if self.numAtoms != self.numRealQMAtoms+self.numRPntChrgs:
                raise ValueError("Unit cell is not complete.")

            cellList = np.genfromtxt(fin, dtype=None, skip_header=1+self.numQMAtoms+self.numPntChrgs,
                                     max_rows=4)
            self.cellOrigin = cellList[0]
            self.cellBasis = cellList[1:4]

    def scale_charges(self, qmSwitchingType=None,
                      qmCutoff=None, qmSwdist=None, **kwargs):
        """Scale external point charges."""
        dij_min2 = self.dij2[0:self.numRPntChrgs, 0:self.numRealQMAtoms].min(axis=1)
        self.dij_min2 = dij_min2
        dij_min_j = self.dij2[0:self.numRPntChrgs, 0:self.numRealQMAtoms].argmin(axis=1)
        self.dij_min_j = dij_min_j
        self.pntDist = np.sqrt(self.dij_min2)

        if qmSwitchingType.lower() == 'shift':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            self.pntScale = (1 - dij_min2/qmCutoff2)**2
            self.pntScale_deriv = 4 * (1 - dij_min2/qmCutoff2) / qmCutoff2
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
        elif qmSwitchingType.lower() == 'switch':
            if qmCutoff is None or qmSwdist is None:
                raise ValueError("We need qmCutoff and qmSwdist here.")
            if qmCutoff <= qmSwdist:
                raise ValueError("qmCutoff should be greater than qmSwdist.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = qmSwdist**2
            self.pntScale = ((dij_min2 - qmCutoff2)**2
                             * (qmCutoff2 + 2*dij_min2 - 3*qmSwdist2)
                             / (qmCutoff2 - qmSwdist2)**3
                             * (dij_min2 >= qmSwdist2)
                             + (dij_min2 < qmSwdist2))
            self.pntScale_deriv = (12 * (dij_min2 - qmSwdist2)
                                   * (qmCutoff2 - dij_min2)
                                   / (qmCutoff2 - qmSwdist2)**3)
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
            self.pntScale_deriv *= (dij_min2 > qmSwdist2)[:, np.newaxis]
        elif qmSwitchingType.lower() == 'lrec':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            scale = 1 - self.pntDist / qmCutoff
            self.pntScale = 1 - (2*scale**3 - 3*scale**2 + 1)**2
            self.pntScale_deriv = 12 * scale * (2*scale**3 - 3*scale**2 + 1) / qmCutoff**2
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
        else:
            raise ValueError("Only 'shift', 'switch', and 'lrec' are supported at the moment.")

        # Just to be safe
        self.pntScale *= (self.pntDist < qmCutoff)
        self.pntScale_deriv *= (self.pntDist < qmCutoff)[:, np.newaxis]

        self.pntScale = np.append(self.pntScale, np.ones(self.numVPntChrgs))
        self.pntChrgsScld = self.pntChrgs * self.pntScale

    def get_qmesp(self):
        """Get electrostatic potential due to external point charges."""
        if self.pbc:
            raise NotImplementedError()
        else:
            self.qmESP = (self.KE * np.sum(self.pntChrgs4QM[:, np.newaxis]
                                        / self.dij, axis=0))
            return self.qmESP

    def get_qmparams(self, calc_forces=None, read_first=False, read_guess=None, addparam=None):
        if calc_forces is not None:
            self.calc_forces = calc_forces
        elif not hasattr(self, 'calc_forces'):
            self.calc_forces = True

        self.read_first = read_first

        if read_guess is not None:
            if self.stepNum == 0 and not self.read_first:
                self.read_guess = False
            else:
                self.read_guess = read_guess
        else:
            self.read_guess = False

        self.addparam = addparam

    def get_nproc(self):
        """Get the number of processes for QM calculation."""
        if 'OMP_NUM_THREADS' in os.environ:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        elif 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1
        return nproc

    def run(self):
        """Run QM calculation."""

        cmdline = self.gen_cmdline()

        if self.stepNum == 0 and not self.read_first:
            self.rm_guess()

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    def corr_elecembed(self):
        """Correct forces due to scaling external point charges in Electrostatic Embedding."""
        if not hasattr(self, 'pntESP'):
            self.get_pntesp()

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

    def corr_vpntchrgs_old(self):
        """Correct forces due to virtual external point charges (deprecated)."""
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                for i in range(self.numMM2):
                    mm1Pos = (self.pntPos[self.numRPntChrgs + i * 3 + 1]
                              - self.pntPos[self.numRPntChrgs + i * 3]
                              * 0.94) / 0.06
                    mm2Pos = self.pntPos[self.numRPntChrgs + i * 3]
                    for j in range(self.numRPntChrgs):
                        if np.abs(mm1Pos - self.pntPos[j]).sum() < 0.001:
                            mm1Idx = j
                        if np.abs(mm2Pos - self.pntPos[j]).sum() < 0.001:
                            mm2Idx = j

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3]
                    self.pntChrgForces[self.numRPntChrgs + i * 3] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] * 0.94
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] * 0.06
                    self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] * 1.06
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] * -0.06
                    self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] = 0.
            elif self.numVPntChrgsPerMM2 == 2:
                raise NotImplementedError()
        else:
            pass

    @classmethod
    def check_software(cls, software):
        return software.lower() == cls.QMTOOL.lower()
