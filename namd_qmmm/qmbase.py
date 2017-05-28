import os
import subprocess as sp
import numpy as np

from . import units

class QMBase(object):

    QMTOOL = None

    def __init__(self, system, charge=None, mult=None, pbc=None):
        """
        Creat a QM object.
        """

        self.basedir = os.path.dirname(system.fin) + "/"
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

        self.n_qm_atoms = system.n_qm_atoms
        self.n_mm_atoms = system.n_mm_atoms
        self.qm_element = system.qm_element[system.map2sorted]
        self.qm_position = system.qm_position[system.map2sorted]
        self.mm_position = system.mm_position
        self.mm_charge_qm = system.mm_charge_qm

        self.rij = system.rij[:, system.map2sorted] / units.L_AU
        self.dij = system.dij[:, system.map2sorted] / units.L_AU
        self.cell_origin = system.cell_origin
        self.cell_basis = system.cell_basis

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

    def get_qm_esp(self):
        """Get electrostatic potential due to external point charges."""
        if self.pbc:
            raise NotImplementedError()
        else:
            self.qm_esp = np.sum(self.mm_charge_qm[:, np.newaxis] / self.dij, axis=0)

            return self.qm_esp

    def get_qm_params(self, calc_forces=None, read_guess=None, addparam=None):
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
