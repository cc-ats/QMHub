import numpy as np

from .mmbase import MMBase
from ..atomtools import QMAtoms, MMAtoms


class OpenMM(MMBase):
    """Class to communicate with NAMD.

    Attributes
    ----------
    n_qm_atoms : int
        Number of QM atoms including linking atoms
    n_mm_atoms : int
        Number of MM atoms including virtual particles
    n_atoms: int
        Number of total atoms in the whole system
    qm_charge : int
        Total charge of QM subsystem
    qm_mult : int
        Multiplicity of QM subsystem
    step : int
        Current step number

    """

    MMTOOL = 'OpenMM'

    def __init__(self, fin):

        self.fin = fin

        self.n_qm_atoms = self.fin.n_qm_atoms
        self.n_mm_atoms = self.fin.n_mm_atoms
        self.n_atoms = self.n_qm_atoms + self.n_mm_atoms

        self.qm_charge = self.fin.qm_charge
        self.qm_mult = self.fin.qm_mult

        self.step = 0

        # Process QM atoms
        qm_pos_x = self.fin.qm_position[:, 0]
        qm_pos_y = self.fin.qm_position[:, 1]
        qm_pos_z = self.fin.qm_position[:, 2]
        qm_element = self.fin.qm_element
        qm_atom_charge = self.fin.qm_atom_charge
        qm_index = self.fin.qm_index
        self.cell_basis = self.fin.cell_basis

        self.qm_atoms = QMAtoms(qm_pos_x, qm_pos_y, qm_pos_z, qm_element,
                                qm_atom_charge, qm_index, self.cell_basis)

        # Process MM atoms
        if self.n_mm_atoms > 0:
            mm_pos_x = self.fin.mm_position[:, 0]
            mm_pos_y = self.fin.mm_position[:, 1]
            mm_pos_z = self.fin.mm_position[:, 2]
            mm_atom_charge = self.fin.mm_atom_charge
            mm_index = self.fin.mm_index

            self.mm_atoms = MMAtoms(mm_pos_x, mm_pos_y, mm_pos_z,
                                    mm_atom_charge, mm_index, mm_atom_charge, self.qm_atoms)
            self.mm_atoms.element = self.fin.mm_element
        else:
            self.mm_atoms = None

    def update_positions(self):
        self.qm_atoms.position = self.fin.qm_position
        self.mm_atoms.position = self.fin.mm_position
        self.step += 1
