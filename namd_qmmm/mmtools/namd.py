import os
import numpy as np

from ..mmbase import MMBase


class NAMD(MMBase):

    MMTOOL = 'NAMD'

    def load_system(self, fin):

        # Open fin file
        f = open(fin, 'r')
        lines = f.readlines()

        # Load system information
        sysList = np.loadtxt(lines[0:1], dtype=int)
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
        qmList = np.loadtxt(lines[1:(1+self.numQMAtoms)],
                            dtype='f8, f8, f8, U2, f8, i8')
        # Positions of QM atoms
        self.qmPos = np.column_stack((qmList['f0'],
                                      qmList['f1'],
                                      qmList['f2']))
        # Elements of QM atoms
        self.qmElmnts = np.char.capitalize(qmList['f3'])
        # Charges of QM atoms
        self.qmChrgs0 = qmList['f4']
        # Indexes of QM atoms
        self.qmIdx = qmList['f5']

        # Number of MM1 atoms which equals to number of linking atoms
        self.numMM1 = np.count_nonzero(self.qmIdx == -1)
        # Number of Real QM atoms
        self.numRealQMAtoms = self.numQMAtoms - self.numMM1

        # Load external point charge information
        if self.numPntChrgs > 0:
            pntList = np.loadtxt(lines[(1+self.numQMAtoms):(1+self.numQMAtoms+self.numPntChrgs)],
                                dtype='f8, f8, f8, f8, i8, i8')
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
        else:
            self.pntPos = None
            self.pntChrgs = None

        # Load unit cell information
        if len(lines) > (1 + self.numQMAtoms + self.numPntChrgs):
            cellList = np.loadtxt(lines[(1+self.numQMAtoms+self.numPntChrgs):(1+self.numQMAtoms+self.numPntChrgs+4)], dtype=float)
            self.cellOrigin = cellList[0]
            self.cellBasis = cellList[1:4]
        else:
            self.cellOrigin = None
            self.cellBasis = None

        # Close fin file
        f.close()

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

    def save_results(self):
        """Save the results of QM calculation to file."""
        if os.path.isfile(self.fin + ".result"):
            os.remove(self.fin + ".result")

        with open(self.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.qmEnergy)
            for i in range(self.numQMAtoms):
                f.write(" ".join(format(j, "22.14e") for j in self.qmForces[i])
                        + "  " + format(self.qmChrgs4MM[i], "22.14e") + "\n")
            for i in range(self.numRPntChrgs):
                f.write(" ".join(format(j, "22.14e") for j in self.pntChrgForces[i]) + "\n")

    def preserve_input(self):
        """Preserve the input file passed from NAMD."""
        import glob
        import shutil
        listInputs = glob.glob(self.fin + "_*")
        if listInputs:
            idx = max([int(i.split('_')[-1]) for i in listInputs]) + 1
        else:
            idx = 0

        if os.path.isfile(self.fin):
            shutil.copyfile(self.fin, self.fin+"_"+str(idx))
