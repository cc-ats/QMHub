import os
import shutil
import numpy as np

from ..qmbase import QMBase
from ..qmtmpl import QMTmpl


class QChem(QMBase):

    QMTOOL = 'Q-Chem'

    def get_qmparams(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(QChem, self).get_qmparams(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for Q-Chem.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for Q-Chem.")

    def gen_input(self):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            jobtype = 'force'
        else:
            jobtype = 'sp'

        if self.read_guess:
            read_guess = 'scf_guess read\n'
        else:
            read_guess = ''

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join(["%s\n" % i for i in self.addparam])
            else:
                addparam = self.addparam + '\n'
        else:
            addparam = ''

        with open(self.baseDir+"qchem.inp", "w") as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(jobtype=jobtype,
                    method=self.method, basis=self.basis,
                    read_guess=read_guess, addparam=addparam))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            for i in range(self.numQMAtoms):
                f.write("".join(["%3s" % self.qmElmnts[i],
                                    "%22.14e" % self.qmPos[i, 0],
                                    "%22.14e" % self.qmPos[i, 1],
                                    "%22.14e" % self.qmPos[i, 2], "\n"]))
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            for i in range(self.numPntChrgs):
                f.write("".join(["%22.14e" % self.pntPos[i, 0],
                                    "%22.14e" % self.pntPos[i, 1],
                                    "%22.14e" % self.pntPos[i, 2],
                                    " %22.14e" % self.pntChrgs4QM[i], "\n"]))
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.baseDir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        if 'QCSCRATCH' in os.environ:
            qmsave = os.environ['QCSCRATCH'] + "/save"
            if os.path.isdir(qmsave):
                shutil.rmtree(qmsave)

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.baseDir + "qchem.out", 'r') as f:
            for line in f:
                line = line.strip().expandtabs()

                if "Charge-charge energy" in line:
                    cc_energy = line.split()[-2]

                if "Total energy" in line:
                    scf_energy = line.split()[-1]
                    break

        self.qmEnergy = float(scf_energy) - float(cc_energy)

        return self.qmEnergy

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""

        self.qmForces = -1 * np.genfromtxt(self.baseDir + "efield.dat",
                                            dtype=float,
                                            skip_header=self.numPntChrgs,
                                            max_rows=self.numQMAtoms)

        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""

        self.pntChrgForces = (np.genfromtxt(self.baseDir + "efield.dat",
                                            dtype=float,
                                            max_rows=self.numPntChrgs)
                                * self.pntChrgs4QM[:, np.newaxis])

        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get Mulliken charges from output of QM calculation."""

        with open(self.baseDir + "qchem.out", 'r') as f:
            for line in f:
                if "Ground-State Mulliken Net Atomic Charges" in line:
                    charges = []
                    for i in range(3):
                        line = next(f)
                    for i in range(self.numQMAtoms):
                        line = next(f)
                        charges.append(float(line.split()[2]))
                    break
        self.qmChrgs = np.array(charges)

        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.pntESP = np.loadtxt(self.baseDir + "esp.dat")

        return self.pntESP
