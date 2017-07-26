import os
import io
import copy
import numpy as np
import pandas as pd

from .mmbase import MMBase
from ..atomtools import QMAtoms, MMAtoms


class NAMD(MMBase):
    """Class to communicate with NAMD.

    Attributes
    ----------
    n_qm_atoms : int
        Number of QM atoms including linking atoms
    n_mm_atoms : int
        Number of MM atoms including virtual particles
    n_atoms: int
        Number of total atoms in the whole system
    step : int
        Current step number
    n_step : int
        Number of total steps to run in the current job

    """

    MMTOOL = 'NAMD'

    def __init__(self, fin):

        self.fin = fin

        # Read fin file
        f = open(self.fin, 'r')
        lines = f.readlines()
        f.close()

        # Load system information
        self.n_qm_atoms, self.n_mm_atoms, self.n_atoms, self.step, self.n_steps = \
            np.fromstring(lines[0], dtype=int, count=5, sep=' ')

        # Load QM information
        f = io.StringIO("".join(lines[1:(self.n_qm_atoms + 1)]))
        qm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=self.n_qm_atoms,
                               names=['pos_x', 'pos_y', 'pos_z', 'element', 'charge', 'idx']).to_records()

        if self.n_mm_atoms > 0:
            f = io.StringIO("".join(lines[(self.n_qm_atoms + 1):(self.n_qm_atoms + self.n_mm_atoms + 1)]))
            mm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=self.n_mm_atoms,
                                   names=['pos_x', 'pos_y', 'pos_z', 'charge', 'idx', 'bonded_to_idx']).to_records()

        # Load unit cell information
        start = 1 + self.n_qm_atoms + self.n_mm_atoms
        stop = start + 4
        cell_list = np.loadtxt(lines[start:stop], dtype=float)
        self.cell_basis = cell_list[0:3]
        self.cell_origin = cell_list[3]

        if np.all(self.cell_basis == 0.0):
            self.cell_basis = None

        # Process QM atoms
        self.n_virt_qm_atoms = np.count_nonzero(qm_atoms.idx == -1)
        self.n_real_qm_atoms = self.n_qm_atoms - self.n_virt_qm_atoms

        # Sort QM atoms
        self._map2sorted = np.concatenate((np.argsort(qm_atoms.idx[0:self.n_real_qm_atoms]),
                                          np.arange(self.n_real_qm_atoms, self.n_qm_atoms)))
        self._map2unsorted = np.argsort(self._map2sorted)

        qm_atoms = qm_atoms[self._map2sorted]

        if np.any(qm_atoms.element.astype(str) == 'nan'):
            qm_element = np.empty(self.n_qm_atoms, dtype=str)
        else:
            qm_element = np.char.capitalize(qm_atoms.element.astype(str))

        # Initialize the QMAtoms object
        self.qm_atoms = QMAtoms(qm_atoms.pos_x, qm_atoms.pos_y, qm_atoms.pos_z,
                                qm_element, qm_atoms.charge, qm_atoms.idx, self.cell_basis)

        # Process MM atoms
        if self.n_mm_atoms > 0:
            self.n_virt_mm_atoms = np.count_nonzero(mm_atoms.idx == -1)
            self.n_real_mm_atoms = self.n_mm_atoms - self.n_virt_mm_atoms

            real_mm_indices = np.s_[0:self.n_real_mm_atoms]
            virt_mm_indices = np.s_[self.n_real_mm_atoms:]

            orig_mm_charge = copy.copy(mm_atoms.charge[real_mm_indices])

            # Prepare for link atoms
            if self.n_virt_mm_atoms > 0:
                # Local indexes of MM1 and QM host atoms
                mm_bonded_to_idx = mm_atoms.bonded_to_idx
                mm1_local_idx = np.where(mm_bonded_to_idx != -1)[0]
                qm_host_local_idx = mm_bonded_to_idx[mm1_local_idx]
                qm_host_local_idx = self._map2unsorted[qm_host_local_idx]

                # Number of MM2 atoms and coefficient to generate the virtual charges
                mm_charge = mm_atoms.charge
                if mm_charge[-1] + mm_charge[-2] < 0.00001:
                    self._mm1_coeff = np.array([0, 0.06, -0.06])
                    self._mm2_coeff = np.array([1, 0.94, 1.06])
                elif mm_charge[-1] + mm_charge[-2] * 2 < 0.00001:
                    self._mm1_coeff = np.array([0, 0.5])
                    self._mm2_coeff = np.array([1, 0.5])
                else:
                    raise ValueError('Something is wrong with point charge alterations.')

                n_mm2 = self.n_virt_mm_atoms // self._mm1_coeff.size

                # Local indexes of MM1 and MM2 atoms the virtual point charges belong to
                mm_pos = np.column_stack([mm_atoms.pos_x, mm_atoms.pos_y, mm_atoms.pos_z])
                virt_mm_pos = mm_pos[virt_mm_indices].reshape(-1, self._mm1_coeff.size, 3)

                virt_atom_mm1_pos = (virt_mm_pos[:, 1] - virt_mm_pos[:, 0] * self._mm2_coeff[1]) / self._mm1_coeff[1]
                virt_atom_mm2_pos = virt_mm_pos[:, 0]

                self._virt_atom_mm1_idx = np.zeros(n_mm2, dtype=int)
                self._virt_atom_mm2_idx = np.zeros(n_mm2, dtype=int)

                for i in range(n_mm2):
                    for j in range(self.n_virt_qm_atoms):
                        if np.abs(virt_atom_mm1_pos[i] - mm_pos[mm1_local_idx[j]]).sum() < 0.001:
                            self._virt_atom_mm1_idx[i] = mm1_local_idx[j]
                            break

                for i in range(n_mm2):
                    for j in range(self.n_real_mm_atoms):
                        if np.abs(virt_atom_mm2_pos[i] - mm_pos[j]).sum() < 0.001:
                            self._virt_atom_mm2_idx[i] = j
                            break

                # Local index of MM2 atoms
                mm2_local_idx = []

                for i in range(self.n_virt_qm_atoms):
                    mm2_local_idx.append(self._virt_atom_mm2_idx[self._virt_atom_mm1_idx == mm1_local_idx[i]])

                # Get original MM charges
                mm1_charge = mm_atoms.charge[virt_mm_indices].reshape(-1, self._mm1_coeff.size).sum(axis=1)
                np.add.at(orig_mm_charge, self._virt_atom_mm1_idx, mm1_charge)

            # Initialize the MMAtoms object
            self.mm_atoms = MMAtoms(mm_atoms.pos_x, mm_atoms.pos_y, mm_atoms.pos_z,
                                    mm_atoms.charge, mm_atoms.idx, orig_mm_charge, self.qm_atoms)

            if self.n_virt_mm_atoms > 0:
                # Get array mask to cancel 1-2 and 1-3 interactions for coulomb
                coulomb_mask = np.ones((self.n_mm_atoms, self.n_qm_atoms), dtype=bool)

                for i in range(self.n_virt_qm_atoms):
                    # Cancel 1-3 interactions between MM2 atoms and QM hosts
                    coulomb_mask[mm2_local_idx[i], qm_host_local_idx[i]] = False
                    # Cancel 1-2 interactions between MM1 atoms and QM hosts
                    coulomb_mask[mm1_local_idx[i], qm_host_local_idx[i]] = False

                # Cancel 1-3 interactions between virtual atoms and QM hosts
                virt_atom_mm1_idx = np.repeat(self._virt_atom_mm1_idx, self._mm1_coeff.size)
                for i in range(self.n_virt_qm_atoms):
                    coulomb_mask[virt_mm_indices][virt_atom_mm1_idx == mm1_local_idx[i], qm_host_local_idx[i]] = False

                self.mm_atoms.coulomb_mask = coulomb_mask

        else:
            self.mm_atoms = None

    def save_results(self):
        """Save the results of QM calculation to file."""
        if os.path.isfile(self.fin + ".result"):
            os.remove(self.fin + ".result")

        qm_force = self.qm_force[self._map2unsorted]
        qm_charge = self.qm_atoms.charge[self._map2unsorted]

        with open(self.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.qm_energy)
            for i in range(self.n_qm_atoms):
                f.write(" ".join(format(j, "22.14e") for j in qm_force[i])
                        + "  " + format(qm_charge[i], "22.14e") + "\n")
            for i in range(self.n_real_mm_atoms):
                f.write(" ".join(format(j, "22.14e") for j in self.mm_force[i]) + "\n")

    def preserve_input(self, n_digits=None):
        """Preserve the input file passed from NAMD."""
        import glob
        import shutil
        prev_inputs = glob.glob(self.fin + "_*")
        if prev_inputs:
            idx = max([int(i.split('_')[-1]) for i in prev_inputs]) + 1
        else:
            idx = 0

        if n_digits is not None:
            idx = '{:0{}}'.format(idx, n_digits)
        else:
            idx = str(idx)

        if os.path.isfile(self.fin):
            shutil.copyfile(self.fin, self.fin + "_" + idx)
