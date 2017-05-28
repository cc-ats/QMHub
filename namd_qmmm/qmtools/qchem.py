import os
import shutil
import numpy as np

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class QChem(QMBase):

    QMTOOL = 'Q-Chem'

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(QChem, self).get_qm_params(**kwargs)

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

        with open(self.basedir + "qchem.inp", 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(jobtype=jobtype,
                    method=self.method, basis=self.basis,
                    read_guess=read_guess, addparam=addparam))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            for i in range(self.n_qm_atoms):
                f.write("".join(["%3s" % self.qm_element[i],
                                    "%22.14e" % self.qm_position[i, 0],
                                    "%22.14e" % self.qm_position[i, 1],
                                    "%22.14e" % self.qm_position[i, 2], "\n"]))
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            for i in range(self.n_mm_atoms):
                f.write("".join(["%22.14e" % self.mm_position[i, 0],
                                    "%22.14e" % self.mm_position[i, 1],
                                    "%22.14e" % self.mm_position[i, 2],
                                    " %22.14e" % self.mm_charge_qm[i], "\n"]))
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        if 'QCSCRATCH' in os.environ:
            qmsave = os.environ['QCSCRATCH'] + "/save"
            if os.path.isdir(qmsave):
                shutil.rmtree(qmsave)

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.basedir + "qchem.out", 'r') as f:
            for line in f:
                line = line.strip().expandtabs()

                if "Charge-charge energy" in line:
                    cc_energy = line.split()[-2]

                if "Total energy" in line:
                    scf_energy = line.split()[-1]
                    break

        self.qm_energy = float(scf_energy) - float(cc_energy)

        return self.qm_energy

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force = -1 * np.genfromtxt(self.basedir + "efield.dat",
                                            dtype=float,
                                            skip_header=self.n_mm_atoms,
                                            max_rows=self.n_qm_atoms)

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        self.mm_force = (np.genfromtxt(self.basedir + "efield.dat",
                                            dtype=float,
                                            max_rows=self.n_mm_atoms)
                                * self.mm_charge_qm[:, np.newaxis])

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""

        with open(self.basedir + "qchem.out", 'r') as f:
            for line in f:
                if "Ground-State Mulliken Net Atomic Charges" in line:
                    charges = []
                    for i in range(3):
                        line = next(f)
                    for i in range(self.n_qm_atoms):
                        line = next(f)
                        charges.append(float(line.split()[2]))
                    break
        self.qm_charge = np.array(charges)

        return self.qm_charge

    def get_mm_esp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.mm_esp = np.loadtxt(self.basedir + "esp.dat")

        return self.mm_esp
