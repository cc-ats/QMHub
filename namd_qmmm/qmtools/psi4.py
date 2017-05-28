from __future__ import absolute_import

import os
import numpy as np

from .qmbase import QMBase

import psi4


class PSI4(QMBase):

    QMTOOL = 'PSI4'

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(PSI4, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for PSI4.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for PSI4.")

    def gen_input(self):
        """Generate input file for QM software."""

        psi4.set_options({'dft_functional': '%s' % self.method,
                          'basis': '%s' % self.basis})

        if self.calc_forces:
            psi4.set_options({'e_convergence': 1e-8,
                              'd_convergence': 1e-8})

        # if self.read_guess:
        #     psi4.set_options({'guess': 'read'})

        if self.addparam is not None:
            addparam = dict(self.addparam)
            psi4.set_options(addparam)

        geom = []
        for i in range(self.n_qm_atoms):
            geom.append("".join(["%3s" % self.qm_element[i],
                                 "%22.14e" % self.qm_position[i, 0],
                                 "%22.14e" % self.qm_position[i, 1],
                                 "%22.14e" % self.qm_position[i, 2], "\n"]))
        geom.append("symmetry c1\n")
        geom = "".join(geom)

        self.molecule = psi4.geometry(geom)
        self.molecule.set_molecular_charge(self.charge)
        self.molecule.set_multiplicity(self.mult)
        self.molecule.fix_com(True)
        self.molecule.fix_orientation(True)

        mm_charge = psi4.QMMM()

        for i in range(self.n_mm_atoms):
            mm_charge.addChargeAngstrom(self.mm_charge_qm[i],
                                       self.mm_position[i, 0],
                                       self.mm_position[i, 1],
                                       self.mm_position[i, 2])
        mm_charge.populateExtern()
        psi4.core.set_global_option_python('EXTERN', mm_charge.extern)

        with open(self.basedir + "grid.dat", 'w') as f:
            for i in range(self.n_mm_atoms):
                f.write("".join(["%22.14e" % self.mm_position[i, 0],
                                 "%22.14e" % self.mm_position[i, 1],
                                 "%22.14e" % self.mm_position[i, 2], "\n"]))

    def run(self):
        """Run QM calculation."""

        nproc = self.get_nproc()
        psi4.core.set_num_threads(nproc, True)

        oldpwd = os.getcwd()
        os.chdir(self.basedir)
        psi4.core.set_output_file(self.basedir + "psi4.out", False)

        # psi4_io = psi4.core.IOManager.shared_object()
        # psi4_io.set_specific_path(32, self.basedir)
        # psi4_io.set_specific_retention(32, True)

        scf_e, self.scf_wfn = psi4.energy('scf', return_wfn=True)

        if self.calc_forces:
            scf_g = psi4.gradient('scf', ref_wfn=self.scf_wfn)

        # psi4.oeprop(self.scf_wfn, 'MULLIKEN_CHARGES', title='SCF')
        self.oeprop = psi4.core.OEProp(self.scf_wfn)
        self.oeprop.add("MULLIKEN_CHARGES")
        self.oeprop.add("GRID_ESP")
        self.oeprop.add("GRID_FIELD")
        self.oeprop.compute()

        os.chdir(oldpwd)

        self.exitcode = 0
        return self.exitcode

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()


    def rm_guess(self):
        """Remove save from previous QM calculation."""

        raise NotImplementedError()

    def get_qm_energy(self):
        """Get QM energy from output of QM calculation."""

        self.qm_energy = self.scf_wfn.energy()

        return self.qm_energy

    def get_qm_force(self):
        """Get QM forces from output of QM calculation."""

        self.qm_force = -1 * np.array(self.scf_wfn.gradient())

        return self.qm_force

    def get_mm_force(self):
        """Get external point charge forces from output of QM calculation."""

        self.mm_force = (np.column_stack([self.oeprop.Exvals(),
                                               self.oeprop.Eyvals(),
                                               self.oeprop.Ezvals()])
                              * self.mm_charge_qm[:, np.newaxis])

        return self.mm_force

    def get_qm_charge(self):
        """Get Mulliken charges from output of QM calculation."""
        self.qm_charge = np.array(self.scf_wfn.atomic_point_charges())

        return self.qm_charge

    def get_mm_esp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.mm_esp = np.array(self.oeprop.Vvals())

        return self.mm_esp
