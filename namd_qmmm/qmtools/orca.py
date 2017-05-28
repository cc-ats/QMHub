import os
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class ORCA(QMBase):

    QMTOOL = 'ORCA'

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(ORCA, self).get_qm_params(**kwargs)

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

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            calc_forces = 'EnGrad '
        else:
            calc_forces = ''

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

        with open(self.basedir + "orca.inp", 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                    method=self.method, basis=self.basis,
                    calc_forces=calc_forces, read_guess=read_guess,
                    addparam=addparam, nproc=nproc,
                    pntchrgspath="\"orca.pntchrg\""))
            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % self.charge)
            f.write("  Mult %d\n" % self.mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(self.n_qm_atoms):
                f.write(" ".join(["%6s" % self.qm_element[i],
                                    "%22.14e" % self.qm_position[i, 0],
                                    "%22.14e" % self.qm_position[i, 1],
                                    "%22.14e" % self.qm_position[i, 2], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        with open(self.basedir + "orca.pntchrg", 'w') as f:
            f.write("%d\n" % self.n_mm_atoms)
            for i in range(self.n_mm_atoms):
                f.write("".join(["%22.14e " % self.mm_charge_qm[i],
                                    "%22.14e" % self.mm_position[i, 0],
                                    "%22.14e" % self.mm_position[i, 1],
                                    "%22.14e" % self.mm_position[i, 2], "\n"]))

        with open(self.basedir + "orca.pntvpot.xyz", 'w') as f:
            f.write("%d\n" % self.n_mm_atoms)
            for i in range(self.n_mm_atoms):
                f.write("".join(["%22.14e" % (self.mm_position[i, 0] / units.L_AU),
                                    "%22.14e" % (self.mm_position[i, 1] / units.L_AU),
                                    "%22.14e" % (self.mm_position[i, 2] / units.L_AU), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.basedir + "; "
        cmdline += "orca orca.inp > orca.out; "
        cmdline += "orca_vpot orca.gbw orca.scfp orca.pntvpot.xyz orca.pntvpot.out >> orca.out"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        qmsave = self.basedir + "orca.gbw"
        if os.path.isfile(qmsave):
            os.remove(qmsave)

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.basedir + "orca.out", 'r') as f:
            for line in f:
                line = line.strip().expandtabs()

                if "FINAL SINGLE POINT ENERGY" in line:
                    self.qm_energy = float(line.split()[-1])
                    break

        return self.qm_energy

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force = -1 * np.genfromtxt(self.basedir + "orca.engrad",
                                    dtype=float, skip_header=11,
                                    max_rows=self.n_qm_atoms*3).reshape((self.n_qm_atoms, 3))

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        self.mm_force = -1 * np.genfromtxt(self.basedir + "orca.pcgrad",
                                                dtype=float,
                                                skip_header=1,
                                                max_rows=self.n_mm_atoms)

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""

        with open(self.basedir + "orca.out", 'r') as f:
            for line in f:
                if "MULLIKEN ATOMIC CHARGES" in line:
                    charges = []
                    line = next(f)
                    for i in range(self.n_qm_atoms):
                        line = next(f)
                        charges.append(float(line.split()[3]))
                    break
        self.qm_charge = np.array(charges)

        return self.qm_charge

    def get_mm_esp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.mm_esp = np.genfromtxt(self.basedir + "orca.pntvpot.out",
                                    dtype=float,
                                    skip_header=1,
                                    usecols=3,
                                    max_rows=self.n_mm_atoms)

        return self.mm_esp
