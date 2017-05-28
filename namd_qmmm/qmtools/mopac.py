from __future__ import division

import os
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class MOPAC(QMBase):

    QMTOOL = 'MOPAC'

    def get_qm_params(self, method=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(MOPAC, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for MOPAC.")

    def gen_input(self):
        """Generate input file for QM software."""

        if not hasattr(self, 'qm_esp'):
            self.get_qm_esp()

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            calc_forces = 'GRAD '
        else:
            calc_forces = ''

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join([" %s" % i for i in self.addparam])
            else:
                addparam = " " + self.addparam
        else:
            addparam = ''

        nproc = self.get_nproc()

        with open(self.basedir + "mopac.mop", 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(method=self.method,
                    charge=self.charge, calc_forces=calc_forces,
                    addparam=addparam, nproc=nproc))
            f.write("NAMD QM/MM\n\n")
            for i in range(self.n_qm_atoms):
                f.write(" ".join(["%6s" % self.qm_element[i],
                                    "%22.14e 1" % self.qm_position[i, 0],
                                    "%22.14e 1" % self.qm_position[i, 1],
                                    "%22.14e 1" % self.qm_position[i, 2], "\n"]))

        with open(self.basedir + "mol.in", 'w') as f:
            f.write("\n")
            f.write("%d %d\n" % (self.n_qm_atoms, 0))

            for i in range(self.n_qm_atoms):
                f.write(" ".join(["%6s" % self.qm_element[i],
                                    "%22.14e" % self.qm_position[i, 0],
                                    "%22.14e" % self.qm_position[i, 1],
                                    "%22.14e" % self.qm_position[i, 2],
                                    " %22.14e" % (self.qm_esp[i] * units.E_AU), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.basedir + "; "
        cmdline += "mopac mopac.mop 2> /dev/null"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        pass

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        with open(self.basedir + "mopac.aux", 'r') as f:
            for line in f:
                if "TOTAL_ENERGY" in line:
                    self.qm_energy = float(line[17:].replace("D", "E")) / units.EH_TO_EV
                    break

        return self.qm_energy

    def get_fij(self):
        """Get pair-wise forces between QM atomic charges and external point charges."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()

        self.fij = (-1 * self.mm_charge_qm[:, np.newaxis] * self.qm_charge[np.newaxis, :]
                  / self.dij**3)
        self.fij = self.fij[:, :, np.newaxis] * self.rij

        return self.fij

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        if not hasattr(self, 'fij'):
            self.get_fij()

        n_lines = int(np.ceil(self.n_qm_atoms * 3 / 10))
        with open(self.basedir + "mopac.aux", 'r') as f:
            for line in f:
                if "GRADIENTS" in line:
                    gradients = np.array([])
                    for i in range(n_lines):
                        line = next(f)
                        gradients = np.append(gradients, np.fromstring(line, sep=' '))
                    break
        self.qm_force = -1 * gradients.reshape(self.n_qm_atoms, 3)
        self.qm_force /= units.F_AU
        self.qm_force -= self.fij.sum(axis=0)

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        if not hasattr(self, 'fij'):
            self.get_fij()

        self.mm_force = self.fij.sum(axis=1)

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""

        n_lines = int(np.ceil(self.n_qm_atoms / 10))
        with open(self.basedir + "mopac.aux", 'r') as f:
            for line in f:
                if "ATOM_CHARGES" in line:
                    charges = np.array([])
                    for i in range(n_lines):
                        line = next(f)
                        charges = np.append(charges, np.fromstring(line, sep=' '))
                    break
        self.qm_charge = charges

        return self.qm_charge

    def get_mm_esp(self):
        """Get ESP at external point charges from output of QM calculation."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()
        self.mm_esp = np.sum(self.qm_charge[np.newaxis, :] / self.dij, axis=1)

        return self.mm_esp
