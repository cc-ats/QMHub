import os
import subprocess as sp
import numpy as np


class QMBase(object):

    QMTOOL = None

    HARTREE2KCALMOL = 6.275094737775374e+02
    HARTREE2EV = 2.721138602e+01
    BOHR2ANGSTROM = 5.2917721067e-01

    def __init__(self, system, charge=None, mult=None, pbc=None):
        """
        Creat a QM object.
        """

        self.baseDir = os.path.dirname(system.fin) + "/"
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

        self.numQMAtoms = system.numQMAtoms
        self.numPntChrgs = system.numPntChrgs
        self.qmElmnts = system.qmElmnts[system.map2sorted]
        self.qmPos = system.qmPos[system.map2sorted]
        self.pntPos = system.pntPos
        self.pntChrgs4QM = system.pntChrgs4QM

        self.rij = system.rij[:, system.map2sorted] / self.BOHR2ANGSTROM
        self.dij = system.dij[:, system.map2sorted] / self.BOHR2ANGSTROM
        self.cellOrigin = system.cellOrigin
        self.cellBasis = system.cellBasis

    @classmethod
    def check_software(cls, software):
        return software.lower() == cls.QMTOOL.lower()

    @staticmethod
    def get_nproc():
        """Get the number of processes for QM calculation."""
        if 'OMP_NUM_THREADS' in os.environ:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        elif 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1
        return nproc

    def get_qmesp(self):
        """Get electrostatic potential due to external point charges."""
        if self.pbc:
            raise NotImplementedError()
        else:
            self.qmESP = np.sum(self.pntChrgs4QM[:, np.newaxis] / self.dij, axis=0)

            return self.qmESP

    def get_qmparams(self, calc_forces=None, read_guess=None, addparam=None):
        if calc_forces is not None:
            self.calc_forces = calc_forces
        elif not hasattr(self, 'calc_forces'):
            self.calc_forces = True

        if read_guess is not None:
            self.read_guess = read_guess
        elif not hasattr(self, 'read_guess'):
            self.read_guess = False

        self.addparam = addparam

    def run(self):
        """Run QM calculation."""

        cmdline = self.gen_cmdline()

        if not self.read_guess:
            self.rm_guess()

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode
