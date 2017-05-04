import os
import sys
import numpy as np
from . import qmtools


class QMMM(object):
    def __init__(self, fin, qmSoftware, qmCharge, qmMult,
                 elecMode, qmElecEmbed=True,
                 qmRefChrgs=None, qmSwitchingType=None,
                 qmCutoff=None, qmSwdist=None, postProc=False):
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

        if self.elecMode.lower() in {'truncation', 'mmewald'}:
            self.qmPBC = False
        elif self.elecMode.lower() == 'qmewald':
            self.qmPBC = True
        else:
            raise ValueError("Only 'truncation', 'mmewald', and 'qmewald' are supported at the moment.")

        self.QM = self.choose_qmtool(fin)

        if qmRefChrgs is not None:
            self.qmRefChrgs = qmRefChrgs
        else:
            self.qmRefChrgs = self.QM.qmChrgs0

        if self.elecMode.lower() == 'qmewald':
            self.QM.qmChrgs4MM = np.zeros(self.QM.numQMAtoms)
        else:
            self.QM.qmChrgs4MM = self.qmRefChrgs

        if self.elecMode.lower() == 'truncation':
            self.qmSwitching = True
        elif self.elecMode.lower() == 'mmewald':
            if self.qmElecEmbed:
                self.qmSwitching = True
            else:
                self.qmSwitching = False
        elif self.elecMode.lower() == 'qmewald':
            self.qmSwitching = False

        if self.qmSwitching:
            if qmSwitchingType is not None:
                self.qmSwitchingType = qmSwitchingType
            else:
                self.qmSwitchingType = 'shift'

            self.qmCutoff = qmCutoff

            if qmSwdist is not None:
                self.qmSwdist = qmSwdist
            else:
                self.qmSwdist = 0.75 * self.qmCutoff

            self.QM.scale_charges(self.qmSwitchingType, self.qmCutoff, self.qmSwdist)

        if self.elecMode.lower() == 'qmewald':
            if not self.qmElecEmbed:
                raise ValueError("Can not use elecMode='qmewald' with qmElecEmbed=False.")
        elif self.elecMode.lower() == 'mmewald':
            self.QM.pntChrgs4MM = self.QM.pntChrgs
            if self.qmElecEmbed:
                self.QM.pntChrgs4QM = self.QM.pntChrgsScld
            else:
                self.QM.pntChrgs4QM = np.zeros(self.QM.numPntChrgs)
        elif self.elecMode.lower() == 'truncation':
            if self.qmElecEmbed:
                self.QM.pntChrgs4QM = self.QM.pntChrgsScld
            else:
                self.QM.pntChrgs4MM = self.QM.pntChrgsScld
                self.QM.pntChrgs4QM = np.zeros(self.QM.numPntChrgs)

        if self.postProc:
            self.QM.calc_forces = False
        else:
            self.QM.calc_forces = True

    def get_namdinput(self):
        """Get the path of NAMD input file (Unfinished)."""
        self.pid = os.popen("ps -p %d -oppid=" % os.getpid()).read().strip()
        self.cmd = os.popen("ps -p %s -ocommand=" % self.pid).read().strip().split()
        self.cwd = os.popen("pwdx %s" % self.pid).read().strip().split()[1]

    def choose_qmtool(self, fin):
        for qmtool in qmtools.__all__:
            qmtool = getattr(qmtools, qmtool)
            if qmtool.check_software(self.qmSoftware):
                return qmtool(fin, self.qmCharge, self.qmMult, self.qmPBC)
        raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac', or 'psi4' for qmSoftware.")

    def run_qm(self, **kwargs):
        """Run QM calculation."""
        self.QM.get_qmparams(**kwargs)
        self.QM.gen_input()
        self.QM.run()
        if self.QM.exitcode != 0:
            sys.exit(self.QM.exitcode)

    def parse_output(self):
        """Parse the output of QM calculation."""
        if hasattr(self.QM, 'exitcode'):
            self.QM.get_qmenergy()
            if not self.postProc:
                self.QM.get_qmforces()
                self.QM.get_pntchrgforces()
                self.QM.get_qmchrgs()
                self.QM.get_pntesp()

                if self.qmElecEmbed and not self.qmPBC:
                    self.QM.corr_elecembed()

                if not self.qmElecEmbed or self.elecMode.lower() == 'mmewald':
                    self.QM.corr_mechembed()

                self.QM.corr_vpntchrgs()
            else:
                self.QM.qmForces = np.zeros((self.QM.numQMAtoms, 3))
                self.QM.pntChrgForces = np.zeros((self.QM.numRPntChrgs, 3))
        else:
            raise ValueError("Need to run_qm() first.")

    def save_results(self):
        """Save the results of QM calculation to file."""
        if hasattr(self.QM, 'qmEnergy'):
            if os.path.isfile(self.QM.fin + ".result"):
                os.remove(self.QM.fin + ".result")

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
        if os.path.isfile(self.QM.fin + ".result"):
            os.remove(self.QM.fin + ".result")

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

        mmScale[self.QM.qmIdx[0:self.QM.numRealQMAtoms]] = np.ones(self.QM.numRealQMAtoms)
        mmChrgs[self.QM.qmIdx[0:self.QM.numRealQMAtoms]] = self.QM.qmChrgs4MM[0:self.QM.numRealQMAtoms]

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
