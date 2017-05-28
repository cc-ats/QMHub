from __future__ import division

import os
import numpy as np

from .. import units
from ..qmbase import QMBase
from ..qmtmpl import QMTmpl

class MOPAC(QMBase):

    QMTOOL = 'MOPAC'

    def get_qmparams(self, method=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(MOPAC, self).get_qmparams(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for MOPAC.")

    def gen_input(self):
        """Generate input file for QM software."""

        if not hasattr(self, 'qmESP'):
            self.get_qmesp()

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            calcforces = 'GRAD '
        else:
            calcforces = ''

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join([" %s" % i for i in self.addparam])
            else:
                addparam = " " + self.addparam
        else:
            addparam = ''

        nproc = self.get_nproc()

        with open(self.baseDir+"mopac.mop", 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(method=self.method,
                    charge=self.charge, calcforces=calcforces,
                    addparam=addparam, nproc=nproc))
            f.write("NAMD QM/MM\n\n")
            for i in range(self.numQMAtoms):
                f.write(" ".join(["%6s" % self.qmElmnts[i],
                                    "%22.14e 1" % self.qmPos[i, 0],
                                    "%22.14e 1" % self.qmPos[i, 1],
                                    "%22.14e 1" % self.qmPos[i, 2], "\n"]))

        with open(self.baseDir+"mol.in", 'w') as f:
            f.write("\n")
            f.write("%d %d\n" % (self.numQMAtoms, 0))

            for i in range(self.numQMAtoms):
                f.write(" ".join(["%6s" % self.qmElmnts[i],
                                    "%22.14e" % self.qmPos[i, 0],
                                    "%22.14e" % self.qmPos[i, 1],
                                    "%22.14e" % self.qmPos[i, 2],
                                    " %22.14e" % (self.qmESP[i] * units.E_AU), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.baseDir + "; "
        cmdline += "mopac mopac.mop 2> /dev/null"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        pass

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.baseDir + "mopac.aux", 'r') as f:
            for line in f:
                if "TOTAL_ENERGY" in line:
                    self.qmEnergy = float(line[17:].replace("D", "E")) / units.EH_TO_EV
                    break

        return self.qmEnergy

    def get_fij(self):
        """Get pair-wise forces between QM atomic charges and external point charges."""

        if not hasattr(self, 'qmChrgs'):
            self.get_qmchrgs()

        self.fij = (-1 * self.pntChrgs4QM[:, np.newaxis] * self.qmChrgs[np.newaxis, :]
                  / self.dij**3)
        self.fij = self.fij[:, :, np.newaxis] * self.rij

        return self.fij

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""

        if not hasattr(self, 'fij'):
            self.get_fij()

        numLines = int(np.ceil(self.numQMAtoms * 3 / 10))
        with open(self.baseDir + "mopac.aux", 'r') as f:
            for line in f:
                if "GRADIENTS" in line:
                    gradients = np.array([])
                    for i in range(numLines):
                        line = next(f)
                        gradients = np.append(gradients, np.fromstring(line, sep=' '))
                    break
        self.qmForces = -1 * gradients.reshape(self.numQMAtoms, 3)
        self.qmForces /= units.F_AU
        self.qmForces -= self.fij.sum(axis=0)

        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""

        if not hasattr(self, 'fij'):
            self.get_fij()

        self.pntChrgForces = self.fij.sum(axis=1)

        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get Mulliken charges from output of QM calculation."""

        numLines = int(np.ceil(self.numQMAtoms / 10))
        with open(self.baseDir + "mopac.aux", 'r') as f:
            for line in f:
                if "ATOM_CHARGES" in line:
                    charges = np.array([])
                    for i in range(numLines):
                        line = next(f)
                        charges = np.append(charges, np.fromstring(line, sep=' '))
                    break
        self.qmChrgs = charges

        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""

        if not hasattr(self, 'qmChrgs'):
            self.get_qmchrgs()
        self.pntESP = np.sum(self.qmChrgs[np.newaxis, :] / self.dij, axis=1)

        return self.pntESP
