import os
import sys
import warnings
import numpy as np
from .qmtool import QM


class QMMM(object):
    def __init__(self, fin, qmSoftware, qmCharge, qmMult,
                 elecMode, qmChargeMode=None, qmElecEmbed='on',
                 qmSwitching=None, qmSwitchingType=None,
                 qmCutoff=None, qmSwdist=None, postProc='no'):
        """
        Creat a QMMM object.
        """
        self.qmSoftware = qmSoftware
        self.qmCharge = qmCharge
        self.qmMult = qmMult
        self.elecMode = elecMode
        self.qmElecEmbed = qmElecEmbed
        self.postProc = postProc

        if self.elecMode.lower() in {'truncation', 'mmewald'}:
            self.qmPBC = 'no'
        elif self.elecMode.lower() == 'qmewald':
            self.qmPBC = 'yes'
        else:
            raise ValueError("Only 'truncation', 'mmewald', and 'qmewald' are supported at the moment.")

        self.QM = QM(fin, self.qmSoftware, self.qmCharge, self.qmMult, self.qmPBC)

        if qmChargeMode is not None:
            self.qmChargeMode = qmChargeMode
        else:
            if self.elecMode.lower() in {'truncation', 'mmewald'}:
                self.qmChargeMode = 'ff'
            elif self.elecMode.lower() == 'qmewald':
                self.qmChargeMode = 'zero'

        if qmSwitching is not None:
            self.qmSwitching = qmSwitching
        else:
            if self.elecMode.lower() in {'truncation', 'mmewald'}:
                self.qmSwitching = 'on'
            elif self.elecMode.lower() == 'qmewald':
                self.qmSwitching = 'off'

        if self.qmSwitching.lower() == 'on':
            if self.elecMode.lower() == 'qmewald':
                raise ValueError("Can not use elecMode='qmewald' with qmSwitching='on'.")
            if qmSwitchingType is not None:
                self.qmSwitchingType = qmSwitchingType
            else:
                self.qmSwitchingType = 'shift'
            self.qmCutoff = qmCutoff
            self.qmSwdist = qmSwdist
            self.QM.scale_charges(self.qmSwitchingType, self.qmCutoff, self.qmSwdist)
        elif self.qmSwitching.lower() == 'off':
            pass
        else:
            raise ValueError("Choose 'on' or 'off' for qmSwitching.")

        if self.qmElecEmbed.lower() == 'on':
            if self.qmSwitching.lower() == 'on':
                self.QM.pntChrgs4QM = self.QM.pntChrgsScld
            elif self.qmSwitching.lower() == 'off':
                self.QM.pntChrgs4QM = self.QM.pntChrgs
                if self.elecMode.lower() != 'qmewald':
                    warnings.warn("There might be discontinuity at the cutoff.")
            if self.elecMode.lower() == 'truncation':
                self.QM.pntChrgs4MM = self.QM.pntChrgs4QM
            elif self.elecMode.lower() == 'mmewald':
                self.QM.pntChrgs4MM = self.QM.pntChrgs
        elif self.qmElecEmbed.lower() == 'off':
            if self.elecMode.lower() == 'qmewald':
                raise ValueError("Can not use elecMode='qmewald' with qmElecEmbed='off'.")
            self.QM.pntChrgs4QM = np.zeros(self.QM.numPntChrgs)
            if self.qmSwitching.lower() == 'on':
                self.QM.pntChrgs4MM = self.QM.pntChrgsScld
                warnings.warn("There might be discontinuity at the cutoff.")
            elif self.qmSwitching.lower() == 'off':
                self.QM.pntChrgs4MM = self.QM.pntChrgs
        else:
            raise ValueError("Choose 'on' or 'off' for qmElecEmbed.")

        if self.postProc.lower() == 'no':
            self.QM.calc_forces = 'yes'
        elif self.postProc.lower() == 'yes':
            self.QM.calc_forces = 'no'
        else:
            raise ValueError("Choose 'yes' or 'no' for postProc.")

    def get_namdinput(self):
        """Get the path of NAMD input file (Unfinished)."""
        self.pid = os.popen("ps -p %d -oppid=" % os.getpid()).read().strip()
        self.cmd = os.popen("ps -p %s -ocommand=" % self.pid).read().strip().split()
        self.cwd = os.popen("pwdx %s" % self.pid).read().strip().split()[1]

    def run_qm(self, **kwargs):
        """Run QM calculation."""
        self.QM.get_qmparams(**kwargs)
        self.QM.run()
        if self.QM.exitcode != 0:
            sys.exit(self.QM.exitcode)

    def parse_output(self):
        """Parse the output of QM calculation."""
        if hasattr(self.QM, 'exitcode'):
            self.QM.get_qmenergy()
            if self.postProc.lower() == 'no':
                self.QM.get_qmforces()
                self.QM.get_pntchrgforces()
                self.QM.get_qmchrgs()
                self.QM.get_pntesp()

                if self.qmElecEmbed.lower() == 'on':
                    if self.qmSwitching.lower() == 'on':
                        self.QM.corr_pntchrgscale()

                if self.qmChargeMode == "qm":
                    self.QM.qmChrgs4MM = self.QM.qmChrgs
                elif self.qmChargeMode == "ff":
                    self.QM.qmChrgs4MM = self.QM.qmChrgs0
                elif self.qmChargeMode == "zero":
                    self.QM.qmChrgs4MM = np.zeros(self.QM.numQMAtoms)
                else:
                    raise ValueError("Please choose 'qm', 'ff', and 'zero' for qmChargeMode.")

                if self.elecMode.lower() in {'truncation', 'mmewald'}:
                    self.QM.corr_qmpntchrgs()

                self.QM.corr_vpntchrgs()
            else:
                self.QM.qmForces = np.zeros((self.QM.numQMAtoms, 3))
                self.QM.pntChrgForces = np.zeros((self.QM.numRPntChrgs, 3))
        else:
            raise ValueError("Need to run_qm() first.")

    def save_results(self):
        """Save the results of QM calculation to file."""
        if hasattr(self.QM, 'qmEnergy'):
            if os.path.isfile(self.QM.fin+".result"):
                os.remove(self.QM.fin+".result")

            with open(self.QM.fin + ".result", 'w') as f:
                f.write("%22.14e\n" % self.QM.qmEnergy)
                for i in range(self.QM.numQMAtoms):
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(self.QM.qmChrgs4MM[i], "22.14e") + "\n")
                for i in range(self.QM.numRPntChrgs):
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.pntChrgForces[i]) + "\n")
        else:
            raise ValueError("Need to parse_output() first.")

    def save_results_old(self):
        """Save the results of QM calculation to file (deprecated)."""
        if os.path.isfile(self.QM.fin+".result"):
            os.remove(self.QM.fin+".result")

        with open(self.QM.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.QM.qmEnergy)
            for i in range(self.QM.numQMAtoms):
                f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                        + "  " + format(self.QM.qmChrgs4MM[i], "22.14e") + "\n")

    def save_extforces(self):
        """Save the MM forces to extforce.dat (deprecated)."""
        if os.path.isfile(self.QM.baseDir + "extforce.dat"):
            os.remove(self.QM.baseDir + "extforce.dat")
        self.mmForces = np.zeros((self.QM.numAtoms, 3))
        self.mmForces[self.QM.pntIdx] = self.QM.pntChrgForces
        with open(self.QM.baseDir + "extforce.dat", 'w') as f:
            for i in range(self.QM.numAtoms):
                f.write("%4d    0  " % (i + 1)
                        + "  ".join(format(j, "22.14e") for j in self.mmForces[i]) +"\n")
            f.write("0.0")

    def save_pntchrgs(self):
        """Save the QM and MM charges to file (for debugging only)."""
        mmScale = np.zeros(self.QM.numAtoms)
        mmDist = np.zeros(self.QM.numAtoms)
        mmChrgs = np.zeros(self.QM.numAtoms)

        if hasattr(self.QM, 'pntScale'):
            mmScale[self.QM.pntIdx[0:self.QM.numRPntChrgs]] = self.QM.pntScale[0:self.QM.numRPntChrgs]
            mmDist[self.QM.pntIdx[0:self.QM.numRPntChrgs]] = self.QM.pntDist
        else:
            mmScale[self.QM.pntIdx[0:self.QM.numRPntChrgs]] += 1
        mmChrgs[self.QM.pntIdx[0:self.QM.numRPntChrgs]] = self.QM.pntChrgs4QM[0:self.QM.numRPntChrgs]

        if self.qmChargeMode == "qm":
            outQMChrgs = self.QM.qmChrgs
        elif self.qmChargeMode == "ff":
            outQMChrgs = self.QM.qmChrgs0
        elif self.qmChargeMode == "zero":
            outQMChrgs = np.zeros(self.QM.numQMAtoms)

        mmScale[self.QM.qmIdx[0:self.QM.numRealQMAtoms]] = np.ones(self.QM.numRealQMAtoms)
        mmChrgs[self.QM.qmIdx[0:self.QM.numRealQMAtoms]] = outQMChrgs[0:self.QM.numRealQMAtoms]

        np.save(self.QM.baseDir + "mmScale", mmScale)
        np.save(self.QM.baseDir + "mmDist", mmDist)
        np.save(self.QM.baseDir + "mmChrgs", mmChrgs)

    def preserve_input(self):
        """Preserve the input file passed from NAMD."""
        import glob
        import shutil
        listInputs = glob.glob(self.QM.fin + "_*")
        if listInputs:
            idx = max([int(i.split('_')[-1]) for i in listInputs]) + 1
        else:
            idx = 0

        if os.path.isfile(self.QM.fin):
            shutil.copyfile(self.QM.fin, self.QM.fin+"_"+str(idx))


if __name__ == "__main__":
    import sys

    qchem = QMMM(sys.argv[1], qmSoftware='qchem', qmCharge=0, qmMult=1,
                 elecMode='mmewald', qmSwitchingType='shift', qmCutoff=12.)
    qchem.run_qm(method='hf', basis='6-31g', pop='pop_mulliken')
    qchem.parse_output()

    dftb = QMMM(sys.argv[1], qmSoftware='dftb+', qmCharge=0, qmMult=1, 
                elecMode='mmewald', qmSwitchingType='shift', qmCutoff=12.)
    dftb.run_qm()
    dftb.parse_output()
