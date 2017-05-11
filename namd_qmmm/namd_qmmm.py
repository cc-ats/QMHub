import sys
import numpy as np

from . import mmtools
from . import qmtools


class QMMM(object):
    def __init__(self, fin, qmSoftware, qmCharge, qmMult,
                 elecMode, qmElecEmbed=True,
                 qmRefChrgs=None, qmSwitchingType=None,
                 qmCutoff=None, qmSwdist=None, 
                 postProc=False, qmReadGuess=False):
        """
        Creat a QMMM object.
        """
        self.qmSoftware = qmSoftware
        self.qmCharge = qmCharge
        self.qmMult = qmMult
        self.elecMode = elecMode
        self.qmElecEmbed = qmElecEmbed
        self.qmRefChrgs = qmRefChrgs
        self.postProc = postProc
        self.qmReadGuess = qmReadGuess

        if self.elecMode.lower() in {'truncation', 'mmewald'}:
            self.qmPBC = False
        elif self.elecMode.lower() == 'qmewald':
            self.qmPBC = True
        else:
            raise ValueError("Only 'truncation', 'mmewald', and 'qmewald' are supported at the moment.")

        self.system = mmtools.NAMD(fin)

        if self.qmPBC:
            if self.system.numAtoms != self.system.numRealQMAtoms + self.system.numRPntChrgs:
                raise ValueError("Unit cell is not complete.")

        if qmRefChrgs is not None:
            self.qmRefChrgs = qmRefChrgs
        else:
            self.qmRefChrgs = self.system.qmChrgs0

        if self.elecMode.lower() == 'qmewald':
            self.system.qmChrgs4MM = np.zeros(self.system.numQMAtoms)
        else:
            self.system.qmChrgs4MM = self.qmRefChrgs

        if self.elecMode.lower() == 'truncation':
            self.qmSwitching = True
        elif self.elecMode.lower() == 'mmewald':
            if self.qmElecEmbed:
                self.qmSwitching = True
            else:
                self.qmSwitching = False
        elif self.elecMode.lower() == 'qmewald':
            self.qmSwitching = False

        self.qmCutoff = qmCutoff
        self.qmSwdist = qmSwdist

        if self.qmSwitching:
            if qmSwitchingType is not None:
                self.qmSwitchingType = qmSwitchingType
            else:
                self.qmSwitchingType = 'shift'

            self.system.get_min_distances()
            self.system.scale_charges(self.qmSwitchingType, self.qmCutoff, self.qmSwdist)

        if self.elecMode.lower() == 'qmewald':
            if not self.qmElecEmbed:
                raise ValueError("Can not use elecMode='qmewald' with qmElecEmbed=False.")
            self.system.pntChrgs4QM = self.system.pntChrgs
        elif self.elecMode.lower() == 'mmewald':
            self.system.pntChrgs4MM = self.system.pntChrgs
            if self.qmElecEmbed:
                self.system.pntChrgs4QM = self.system.pntChrgsScld
            else:
                self.system.pntChrgs4QM = np.zeros(self.system.numPntChrgs)
        elif self.elecMode.lower() == 'truncation':
            if self.qmElecEmbed:
                self.system.pntChrgs4QM = self.system.pntChrgsScld
            else:
                self.system.pntChrgs4MM = self.system.pntChrgsScld
                self.system.pntChrgs4QM = np.zeros(self.system.numPntChrgs)

        self.qm = self.choose_qmtool()

        if self.postProc:
            self.qm.calc_forces = False
        else:
            self.qm.calc_forces = True

        if self.qmReadGuess and not self.stepNum == 0:
            self.qm.read_guess = True
        else:
            self.qm.read_guess = False

    def choose_qmtool(self):
        for qmtool in qmtools.__all__:
            qmtool = getattr(qmtools, qmtool)
            if qmtool.check_software(self.qmSoftware):
                return qmtool(self.system, self.qmCharge, self.qmMult, self.qmPBC)
        raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac', or 'psi4' for qmSoftware.")

    def run_qm(self, **kwargs):
        """Run QM calculation."""
        self.qm.get_qmparams(**kwargs)
        self.qm.gen_input()
        self.qm.run()
        if self.qm.exitcode != 0:
            sys.exit(self.qm.exitcode)

    def parse_output(self):
        """Parse the output of QM calculation."""
        if hasattr(self.qm, 'exitcode'):
            self.system.parse_output(self.qm)
            if not self.postProc:
                if self.qmElecEmbed and not self.qmPBC:
                    self.system.corr_elecembed()

                if not self.qmElecEmbed or self.elecMode.lower() == 'mmewald':
                    self.system.corr_mechembed()

                self.system.corr_vpntchrgs()
        else:
            raise ValueError("Need to run_qm() first.")

    def save_results(self):
        """Save the results of QM calculation to file."""
        if hasattr(self.system, 'qmEnergy'):
            self.system.save_results()
        else:
            raise ValueError("Need to parse_output() first.")

    def save_pntchrgs(self):
        """Save the QM and MM charges to file (for debugging only)."""
        mmScale = np.zeros(self.system.numAtoms)
        mmDist = np.zeros(self.system.numAtoms)
        mmChrgs = np.zeros(self.system.numAtoms)

        if hasattr(self.system, 'pntScale'):
            mmScale[self.system.pntIdx[0:self.system.numRPntChrgs]] = self.system.pntScale
            mmDist[self.system.pntIdx[0:self.system.numRPntChrgs]] = self.system.dij_min
        else:
            mmScale[self.system.pntIdx[0:self.system.numRPntChrgs]] += 1
        mmChrgs[self.system.pntIdx[0:self.system.numRPntChrgs]] = self.system.pntChrgs4QM[0:self.system.numRPntChrgs]

        mmScale[self.system.qmIdx[0:self.system.numRealQMAtoms]] = np.ones(self.system.numRealQMAtoms)
        mmChrgs[self.system.qmIdx[0:self.system.numRealQMAtoms]] = self.system.qmChrgs4MM[0:self.system.numRealQMAtoms]

        np.save(self.qm.baseDir + "mmScale", mmScale)
        np.save(self.qm.baseDir + "mmDist", mmDist)
        np.save(self.qm.baseDir + "mmChrgs", mmChrgs)

    def preserve_input(self):
        """Preserve the input file passed from NAMD."""
        import glob
        import os
        import shutil
        listInputs = glob.glob(self.system.fin + "_*")
        if listInputs:
            idx = max([int(i.split('_')[-1]) for i in listInputs]) + 1
        else:
            idx = 0

        if os.path.isfile(self.system.fin):
            shutil.copyfile(self.system.fin, self.system.fin+"_"+str(idx))
