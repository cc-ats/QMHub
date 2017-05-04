from __future__ import absolute_import

import os
import numpy as np

from ..qmbase import QMBase
from ..qmtmplt import QMTmplt

import psi4

class PSI4(QMBase):

    SOFTWARE = 'PSI4'

    def get_qmparams(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(self.__class__, self).get_qmparams(**kwargs)

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
        for i in range(self.numQMAtoms):
            geom.append("".join(["%3s" % self.qmElmntsSorted[i],
                                 "%22.14e" % self.qmPosSorted[i, 0],
                                 "%22.14e" % self.qmPosSorted[i, 1],
                                 "%22.14e" % self.qmPosSorted[i, 2], "\n"]))
        geom.append("symmetry c1\n")
        geom = "".join(geom)

        self.molecule = psi4.geometry(geom)
        self.molecule.set_molecular_charge(self.charge)
        self.molecule.set_multiplicity(self.mult)
        self.molecule.fix_com(True)
        self.molecule.fix_orientation(True)

        pntChrgs = psi4.QMMM()

        for i in range(self.numPntChrgs):
            pntChrgs.addChargeAngstrom(self.pntChrgs4QM[i],
                                       self.pntPos[i, 0],
                                       self.pntPos[i, 1],
                                       self.pntPos[i, 2])
        pntChrgs.populateExtern()
        psi4.core.set_global_option_python('EXTERN', pntChrgs.extern)

        with open(self.baseDir+"grid.dat", "w") as f:
            for i in range(self.numPntChrgs):
                f.write("".join(["%22.14e" % self.pntPos[i, 0],
                                 "%22.14e" % self.pntPos[i, 1],
                                 "%22.14e" % self.pntPos[i, 2], "\n"]))

    def run(self):
        """Run QM calculation."""

        nproc = self.get_nproc()
        psi4.core.set_num_threads(nproc, True)

        oldpwd = os.getcwd()
        os.chdir(self.baseDir)
        psi4.core.set_output_file(self.baseDir + "psi4.out", False)

        # psi4_io = psi4.core.IOManager.shared_object()
        # psi4_io.set_specific_path(32, self.baseDir)
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

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""

        self.qmEnergy = self.scf_wfn.energy()
        self.qmEnergy *= self.HARTREE2KCALMOL

        return self.qmEnergy

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""

        self.qmForces = -1 * np.array(self.scf_wfn.gradient())
        self.qmForces *= self.HARTREE2KCALMOL / self.BOHR2ANGSTROM

        # Unsort QM atoms
        self.qmForces = self.qmForces[self.map2unsorted]

        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""

        self.pntChrgForces = (np.column_stack([self.oeprop.Exvals(),
                                               self.oeprop.Eyvals(),
                                               self.oeprop.Ezvals()])
                              * self.pntChrgs4QM[:, np.newaxis])
        self.pntChrgForces *= self.HARTREE2KCALMOL / self.BOHR2ANGSTROM

        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get Mulliken charges from output of QM calculation."""
        self.qmChrgs = np.array(self.scf_wfn.atomic_point_charges())

        # Unsort QM atoms
        self.qmChrgs = self.qmChrgs[self.map2unsorted]

        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""

        self.pntESP = np.array(self.oeprop.Vvals())
        self.pntESP *= self.HARTREE2KCALMOL

        return self.pntESP
