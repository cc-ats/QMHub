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
        self.qmCutoff = qmCutoff
        self.qmSwdist = qmSwdist
        self.postProc = postProc
        self.qmReadGuess = qmReadGuess

        # Initialize the system
        self.system = mmtools.NAMD(fin)

        # Set the refernce charges for QM atoms
        if qmRefChrgs is not None:
            self.qmRefChrgs = qmRefChrgs
        else:
            self.qmRefChrgs = self.system.qmChrgs0

        # Prepare for the electrostatic model
        if self.elecMode.lower() in {'truncation', 'mmewald'}:
            self.qmPBC = False
            self.system.qmChrgs4MM = self.qmRefChrgs
        elif self.elecMode.lower() == 'qmewald':
            self.qmPBC = True
            self.system.qmChrgs4MM = np.zeros(self.system.numQMAtoms)
        else:
            raise ValueError("Only 'truncation', 'mmewald', and 'qmewald' are supported at the moment.")

        # Prepare for QM with PBC
        if self.qmPBC:
            if self.system.numAtoms != self.system.numRealQMAtoms + self.system.numRPntChrgs:
                raise ValueError("Unit cell is not complete.")

            self.split_pntchrgs(qmCutoff=self.qmCutoff)

        # Set switching function for external point charges
        if qmSwitchingType is not None:
            self.qmSwitchingType = qmSwitchingType
        else:
            self.qmSwitchingType = 'shift'

        if self.elecMode.lower() == 'mmewald':
            if not self.qmElecEmbed:
                self.qmSwitchingType = None
        elif self.elecMode.lower() == 'qmewald':
            self.qmSwitchingType = None

        # Scale the external point charges
        if self.qmSwitchingType is not None:
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

        # Initialize the QM system
        self.qm = qmtools.choose_qmtool(self.qmSoftware)(self.system, self.qmCharge, self.qmMult, self.qmPBC)

        if self.postProc:
            self.qm.calc_forces = False
        else:
            self.qm.calc_forces = True

        if self.qmReadGuess and not self.stepNum == 0:
            self.qm.read_guess = True
        else:
            self.qm.read_guess = False

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
            if self.qm.exitcode == 0:
                self.system.parse_output(self.qm)
                if not self.postProc:
                    if self.qmElecEmbed and not self.qmPBC:
                        self.system.corr_elecembed()

                    if not self.qmElecEmbed or self.elecMode.lower() == 'mmewald':
                        self.system.corr_mechembed()

                    self.system.corr_vpntchrgs()
            else:
                raise ValueError("QM calculation did not finish normally.")
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
