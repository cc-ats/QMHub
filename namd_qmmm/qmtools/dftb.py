import os
import numpy as np

from ..qmbase import QMBase
from ..qmtmpl import QMTmpl


class DFTB(QMBase):

    QMTOOL = 'DFTB+'

    def get_qm_params(self, skfpath=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(DFTB, self).get_qm_params(**kwargs)

        if skfpath is not None:
            self.skfpath = os.path.join(skfpath, '')
        else:
            raise ValueError("Please set skfpath for DFTB+.")

    def gen_input(self):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        elements = np.unique(self.qm_element).tolist()
        MaxAngularMomentum = "\n    ".join([i+" = "+qmtmpl.MaxAngularMomentum[i] for i in elements])
        HubbardDerivs = "\n    ".join([i+" = "+qmtmpl.HubbardDerivs[i] for i in elements])

        if self.pbc:
            KPointsAndWeights = qmtmpl.KPointsAndWeights
        else:
            KPointsAndWeights = ""

        if self.calc_forces:
            calc_forces = 'Yes'
        else:
            calc_forces = 'No'

        if self.read_guess:
            read_guess = 'Yes'
        else:
            read_guess = 'No'

        if self.addparam is not None:
            addparam = self.addparam
        else:
            addparam = ''

        with open(self.basedir + "dftb_in.hsd", 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(charge=self.charge,
                    n_mm_atoms=self.n_mm_atoms, read_guess=read_guess,
                    calc_forces=calc_forces, skfpath=self.skfpath,
                    MaxAngularMomentum=MaxAngularMomentum,
                    HubbardDerivs=HubbardDerivs,
                    KPointsAndWeights=KPointsAndWeights,
                    addparam=addparam))
        with open(self.basedir + "input_geometry.gen", 'w') as f:
            if self.pbc:
                f.write(str(self.n_qm_atoms) + " S" + "\n")
            else:
                f.write(str(self.n_qm_atoms) + " C" + "\n")
            f.write(" ".join(elements) + "\n")
            for i in range(self.n_qm_atoms):
                f.write("".join(["%6d" % (i+1),
                                    "%4d" % (elements.index(self.qm_element[i])+1),
                                    "%22.14e" % self.qm_position[i, 0],
                                    "%22.14e" % self.qm_position[i, 1],
                                    "%22.14e" % self.qm_position[i, 2], "\n"]))
            if self.pbc:
                f.write("".join(["%22.14e" % i for i in self.cell_origin]) + "\n")
                f.write("".join(["%22.14e" % i for i in self.cell_basis[0]]) + "\n")
                f.write("".join(["%22.14e" % i for i in self.cell_basis[1]]) + "\n")
                f.write("".join(["%22.14e" % i for i in self.cell_basis[2]]) + "\n")

        with open(self.basedir + "charges.dat", 'w') as f:
            for i in range(self.n_mm_atoms):
                f.write("".join(["%22.14e" % self.mm_position[i, 0],
                                    "%22.14e" % self.mm_position[i, 1],
                                    "%22.14e" % self.mm_position[i, 2],
                                    " %22.14e" % self.mm_charge_qm[i], "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "export OMP_NUM_THREADS=%d; dftb+ > dftb.out" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        qmsave = self.basedir + "charges.bin"
        if os.path.isfile(qmsave):
            os.remove(qmsave)

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        self.qm_energy = np.genfromtxt(self.basedir + "results.tag",
                                        dtype=float, skip_header=1,
                                        max_rows=1)

        return self.qm_energy

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force = np.genfromtxt(self.basedir + "results.tag",
                                        dtype=float, skip_header=5,
                                        max_rows=self.n_qm_atoms)

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        self.mm_force = np.genfromtxt(self.basedir + "results.tag",
                                            dtype=float,
                                            skip_header=self.n_qm_atoms+6,
                                            max_rows=self.n_mm_atoms)

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""

        if self.n_qm_atoms > 3:
            self.qm_charge = np.genfromtxt(
                self.basedir + "results.tag", dtype=float,
                skip_header=(self.n_qm_atoms + self.n_mm_atoms
                                + int(np.ceil(self.n_qm_atoms/3.)) + 14),
                max_rows=int(np.ceil(self.n_qm_atoms/3.)-1.))
        else:
            self.qm_charge = np.array([])
        self.qm_charge = np.append(
            self.qm_charge.flatten(),
            np.genfromtxt(self.basedir + "results.tag", dtype=float,
                skip_header=(self.n_qm_atoms + self.n_mm_atoms
                                + int(np.ceil(self.n_qm_atoms/3.))*2 + 13),
                max_rows=1).flatten())

        return self.qm_charge

    def get_mm_esp(self):
        """Get ESP at external point charges from output of QM calculation."""

        if not hasattr(self, 'qm_charge'):
            self.get_qm_charge()
        self.mm_esp = np.sum(self.qm_charge[np.newaxis, :] / self.dij, axis=1)

        return self.mm_esp
