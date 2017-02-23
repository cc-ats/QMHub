#!/usr/bin/env python3
import os
import numpy as np
from .qmtool import QM

class QMMM(object):
    def __init__(self, fin=None, qmBondScheme='CS', qmElecEmbed='on',
                 qmSwitching='off', qmSwitchingType='shift', qmSoftware=None,
                 qmChargeMode="qm", qmCharge=None, qmMult=None,
                 cutoff=None, swdist=None, PME='no', numAtoms=None):
        """
        Creat a QMMM object.
        """
        self.qmBondScheme = qmBondScheme
        self.qmElecEmbed = qmElecEmbed
        self.qmSwitching = qmSwitching
        self.qmSwitchingType = qmSwitchingType
        self.qmSoftware = qmSoftware
        self.qmChargeMode = qmChargeMode
        self.qmCharge = qmCharge
        self.qmMult = qmMult
        self.PME = PME
        self.numAtoms = numAtoms

        self.QM = QM(fin, self.qmSoftware, self.qmCharge, self.qmMult)

        self.QM.get_dij2(self.qmBondScheme)

        if self.qmElecEmbed.lower() == 'off':
            self.QM.zero_pntChrgs()
        elif self.qmElecEmbed.lower() == 'on':
            if self.qmSwitching.lower() == 'on':
                if self.qmSwitchingType.lower() == 'shift':
                    self.cutoff = cutoff
                    self.QM.scale_charges(self.qmSwitchingType, self.cutoff)
                elif self.qmSwitchingType.lower() == 'switch':
                    self.cutoff = cutoff
                    self.swdist = swdist
                    self.QM.scale_charges(self.qmSwitchingType, self.cutoff, self.swdist)
                else:
                    raise ValueError("Only 'shift' and 'switch' are supported currently.")
        else:
            raise ValueError("We need a valid value for 'qmElecEmbed'.")

    def run_qm(self, **kwargs):
        self.QM.get_qmparams(**kwargs)
        return self.QM.run()

    def parse_output(self):
        self.QM.get_qmenergy()
        self.QM.get_qmforces()
        self.QM.get_pntchrgforces()
        self.QM.get_qmchrgs()
        self.QM.get_pntesp()

        if self.qmSwitching.lower() == 'on':
            self.QM.corr_pntchrgscale()

        if self.PME.lower() == 'yes':
            self.QM.corr_pbc()

        self.QM.corr_vpntchrgs()

    def save_results(self):
        if os.path.isfile(self.QM.fin+".result"):
            os.remove(self.QM.fin+".result")
        with open(self.QM.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.QM.qmEnergy)
            for i in range(self.QM.numQMAtoms):
                if self.qmChargeMode == "qm":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(self.QM.qmChrgs[i], "22.14e") + "\n")
                elif self.qmChargeMode == "ff":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(self.QM.qmChrgs0[i], "22.14e") + "\n")
                elif self.qmChargeMode == "zero":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(0., "22.14e") + "\n")
            for i in range(self.QM.numRPntChrgs):
                f.write(" ".join(format(j, "22.14e") for j in self.QM.pntChrgForces[i]) + "\n")

    def save_results_old(self):
        if os.path.isfile(self.QM.fin+".result"):
            os.remove(self.QM.fin+".result")
        with open(self.QM.fin + ".result", 'w') as f:
            f.write("%22.14e\n" % self.QM.qmEnergy)
            for i in range(self.QM.numQMAtoms):
                if self.qmChargeMode == "qm":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(self.QM.qmChrgs[i], "22.14e") + "\n")
                elif self.qmChargeMode == "ff":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(self.QM.qmChrgs0[i], "22.14e") + "\n")
                elif self.qmChargeMode == "zero":
                    f.write(" ".join(format(j, "22.14e") for j in self.QM.qmForces[i])
                            + "  " + format(0., "22.14e") + "\n")

    def save_extforces(self):
        if os.path.isfile(self.QM.baseDir + "extforce.dat"):
            os.remove(self.QM.baseDir + "extforce.dat")
        self.mmForces = np.zeros((self.numAtoms, 3))
        self.mmForces[self.QM.pntIdx] = self.QM.pntChrgForces
        with open(self.QM.baseDir + "extforce.dat", 'w') as f:
            for i in range(self.numAtoms):
                f.write("%4d    0  " % (i + 1)
                        + "  ".join(format(j, "22.14e") for j in self.mmForces[i]) +"\n")
            f.write("0.0")

    def save_pntchrgs(self):
        mmScale = np.zeros(self.numAtoms)
        mmDist = np.zeros(self.numAtoms)
        mmChrgs = np.zeros(self.numAtoms)

        mmScale[pntIdx[0:self.QM.numRPntChrgs]] = self.QM.pntScale[0:self.QM.numRPntChrgs]
        mmDist[pntIdx[0:self.QM.numRPntChrgs]] = self.QM.pntDist
        mmChrgs[pntIdx[0:self.QM.numRPntChrgs]] = self.QM.outPntChrgs[0:self.QM.numRPntChrgs]

        np.save(self.QM.baseDir + "mmScale", mmScale)
        np.save(self.QM.baseDir + "mmDist", mmDist)
        np.save(self.QM.baseDir + "mmChrgs", mmChrgs)

if __name__ == "__main__":
    import sys

    qchem = QMMM(sys.argv[1], qmSwitching='on', qmSoftware='qchem',
                 qmChargeMode='ff', qmCharge=0, qmMult=1, cutoff=12.,
                 PME='yes', numAtoms=2279)
    qchem.run_qm(method='hf', basis='6-31g',
                 scf_guess='sad', pop='pop_mulliken')
    qchem.parse_output()

    dftb = QMMM(sys.argv[1], qmSwitching='on', qmSoftware='dftb+',
                qmChargeMode='ff', qmCharge=0, qmMult=1, cutoff=12.,
                PME='yes', numAtoms=2279)
    dftb.run_qm(initial_scc='No')
    dftb.parse_output()

