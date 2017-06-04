import os
import copy
import numpy as np

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
            np.loadtxt(lines[0:1], dtype=int, unpack=True)

        # Load QM information
        qm_atoms = np.loadtxt(lines[1:(self.n_qm_atoms + 1)], delimiter=' ',
                              dtype=[('position', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
                                     ('element', 'S2'), ('charge', 'f8'), ('index', 'i8')])
        qm_atoms = qm_atoms.view(np.recarray)

        self.n_virt_qm_atoms = np.count_nonzero(qm_atoms.index == -1)
        self.n_real_qm_atoms = self.n_qm_atoms - self.n_virt_qm_atoms

        # Sort QM atoms
        self._map2sorted = np.concatenate((np.argsort(qm_atoms.index[0:self.n_real_qm_atoms]),
                                          np.arange(self.n_real_qm_atoms, self.n_qm_atoms)))
        self._map2unsorted = np.argsort(self._map2sorted)

        qm_atoms = qm_atoms[self._map2sorted]
        qm_element = np.core.defchararray.decode(np.char.capitalize(qm_atoms.element))

        # Initialize the QMAtoms object
        self.qm_atoms = QMAtoms(qm_atoms.position.x, qm_atoms.position.y, qm_atoms.position.z,
                                qm_element, qm_atoms.charge, qm_atoms.index)

        # Load unit cell information
        start = 1 + self.n_qm_atoms + self.n_mm_atoms
        stop = start + 4
        cell_list = np.loadtxt(lines[start:stop], dtype=float)
        self.cell_basis = cell_list[0:3]
        self.cell_origin = cell_list[3]

        # Load MM information
        if self.n_mm_atoms > 0:
            mm_atoms = np.loadtxt(lines[(1+self.n_qm_atoms):(1+self.n_qm_atoms+self.n_mm_atoms)],
                                  dtype=[('position', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
                                         ('charge', 'f8'), ('index', 'i8'), ('bonded_to_idx', 'i8')])
            mm_atoms = mm_atoms.view(np.recarray)

            self.n_virt_mm_atoms = np.count_nonzero(mm_atoms.index == -1)
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
                mm_pos = mm_atoms.position.view((float, 3))
                virt_mm_pos = mm_atoms.position.view((float, 3))[virt_mm_indices].reshape(-1, self._mm1_coeff.size, 3)

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
            self.mm_atoms = MMAtoms(mm_atoms.position.x, mm_atoms.position.y, mm_atoms.position.z,
                                    mm_atoms.charge, mm_atoms.index, orig_mm_charge, self.qm_atoms)

            if self.n_virt_mm_atoms > 0:
                # Get array mask to cancel 1-2 and 1-3 interactions for coulomb
                coulomb_mask = np.ones((self.n_mm_atoms, self.n_qm_atoms), dtype=bool)

                for i in range(self.n_virt_qm_atoms):
                    # Cancel 1-3 interactions between MM2 atoms and QM hosts
                    coulomb_mask[mm2_local_idx[i], qm_host_local_idx[i]] = False
                    # Cancel 1-2 interactions between MM1 atoms and QM hosts
                    coulomb_mask[mm1_local_idx[i], qm_host_local_idx[i]] = False

                # Cancel 1-3 interactions between virtual atoms and QM hosts
                start = 0
                for i in range(self.n_virt_qm_atoms):
                    stop = start + mm2_local_idx[i].size * self._mm1_coeff.size
                    coulomb_mask[virt_mm_indices][start:stop, qm_host_local_idx[i]] = False
                    start = stop

                self.mm_atoms.coulomb_mask = coulomb_mask

        else:
            self.mm_atoms = None

    def save_results(self):
        """Save the results of QM calculation to file."""
        if os.path.isfile(self.fin + ".result"):
            os.remove(self.fin + ".result")

        qm_force = self.qm_force[self._map2unsorted]
        qm_charge_me = self.qm_charge_me[self._map2unsorted]

        with open(self.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.qm_energy)
            for i in range(self.n_qm_atoms):
                f.write(" ".join(format(j, "22.14e") for j in qm_force[i])
                        + "  " + format(qm_charge_me[i], "22.14e") + "\n")
            for i in range(self.n_real_mm_atoms):
                f.write(" ".join(format(j, "22.14e") for j in self.mm_force[i]) + "\n")

    def preserve_input(self):
        """Preserve the input file passed from NAMD."""
        import glob
        import shutil
        prev_inputs = glob.glob(self.fin + "_*")
        if prev_inputs:
            idx = max([int(i.split('_')[-1]) for i in prev_inputs]) + 1
        else:
            idx = 0

        if os.path.isfile(self.fin):
            shutil.copyfile(self.fin, self.fin+"_"+str(idx))
