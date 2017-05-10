import os
import numpy as np

from ..qmbase import QMBase
from ..qmtmplt import QMTmplt

class ORCA(QMBase):

    QMTOOL = 'ORCA'

    def get_qmparams(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(ORCA, self).get_qmparams(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for ORCA.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for ORCA.")

    def gen_input(self):
        """Generate input file for QM software."""

        qmtmplt = QMTmplt(self.QMTOOL, self.pbc)

        if self.calc_forces:
            calcforces = 'EnGrad '
        else:
            calcforces = ''

        if self.read_guess:
            read_guess = ''
        else:
            read_guess = 'NoAutoStart '

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join(["%s " % i for i in self.addparam])
            else:
                addparam = self.addparam + " "
        else:
            addparam = ''

        nproc = self.get_nproc()

        with open(self.baseDir + "orca.inp", 'w') as f:
            f.write(qmtmplt.gen_qmtmplt().substitute(
                    method=self.method, basis=self.basis,
                    calcforces=calcforces, read_guess=read_guess,
                    addparam=addparam, nproc=nproc,
                    pntchrgspath="\"orca.pntchrg\""))
            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % self.charge)
            f.write("  Mult %d\n" % self.mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(self.numQMAtoms):
                f.write(" ".join(["%6s" % self.qmElmnts[i],
                                    "%22.14e" % self.qmPos[i, 0],
                                    "%22.14e" % self.qmPos[i, 1],
                                    "%22.14e" % self.qmPos[i, 2], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        with open(self.baseDir + "orca.pntchrg", 'w') as f:
            f.write("%d\n" % self.numPntChrgs)
            for i in range(self.numPntChrgs):
                f.write("".join(["%22.14e " % self.pntChrgs4QM[i],
                                    "%22.14e" % self.pntPos[i, 0],
                                    "%22.14e" % self.pntPos[i, 1],
                                    "%22.14e" % self.pntPos[i, 2], "\n"]))

        with open(self.baseDir + "orca.pntvpot.xyz", 'w') as f:
            f.write("%d\n" % self.numPntChrgs)
            for i in range(self.numPntChrgs):
                f.write("".join(["%22.14e" % (self.pntPos[i, 0] / self.BOHR2ANGSTROM),
                                    "%22.14e" % (self.pntPos[i, 1] / self.BOHR2ANGSTROM),
                                    "%22.14e" % (self.pntPos[i, 2] / self.BOHR2ANGSTROM), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.baseDir + "; "
        cmdline += "orca orca.inp > orca.out; "
        cmdline += "orca_vpot orca.gbw orca.scfp orca.pntvpot.xyz orca.pntvpot.out >> orca.out"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        qmsave = self.baseDir + "orca.gbw"
        if os.path.isfile(qmsave):
            os.remove(qmsave)

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.baseDir + "orca.out", 'r') as f:
            for line in f:
                line = line.strip().expandtabs()

                if "FINAL SINGLE POINT ENERGY" in line:
                    self.qmEnergy = float(line.split()[-1])
                    break

        return self.qmEnergy

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""

        self.qmForces = -1 * np.genfromtxt(self.baseDir + "orca.engrad",
                                    dtype=float, skip_header=11,
                                    max_rows=self.numQMAtoms*3).reshape((self.numQMAtoms, 3))

        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""

        self.pntChrgForces = -1 * np.genfromtxt(self.baseDir + "orca.pcgrad",
                                                dtype=float,
                                                skip_header=1,
                                                max_rows=self.numPntChrgs)

        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get Mulliken charges from output of QM calculation."""

        with open(self.baseDir + "orca.out", 'r') as f:
            for line in f:
                if "MULLIKEN ATOMIC CHARGES" in line:
                    charges = []
                    line = next(f)
                    for i in range(self.numQMAtoms):
                        line = next(f)
                        charges.append(float(line.split()[3]))
                    break
        self.qmChrgs = np.array(charges)

        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.pntESP = np.genfromtxt(self.baseDir + "orca.pntvpot.out",
                                    dtype=float,
                                    skip_header=1,
                                    usecols=3,
                                    max_rows=self.numPntChrgs)

        return self.pntESP
