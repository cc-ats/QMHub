import os
import shutil
import subprocess as sp
import numpy as np
from .qmtmplt import QMTmplt

hartree2kcalmol = 6.275094737775374e+02
bohr2angstrom = 5.2917721067e-01
ke = hartree2kcalmol * bohr2angstrom

class QM(object):
    def __init__(self, fin, software=None, charge=None, mult=None):
        """
        Creat a QM object.
        """
        if fin is not None:
            self.fin = os.path.abspath(fin)
        else:
            raise ValueError("We need the QMMM file passed from NAMD.")
        if software is not None:
            self.software = software
        else:
            raise ValueError("Please choose 'qmSoftware' from 'qchem' and 'dftb+'.")
        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for Q-Chem.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1

        numsList = np.genfromtxt(fin, dtype=int, max_rows=1, unpack=True)

        # Number of QM atoms including linking atoms
        self.numQMAtoms = numsList[0]
        # Number of external external point charges including virtual particles
        self.numPntChrgs = numsList[1]
        # Number of current step
        self.stepNum = numsList[2]
        # Number of total steps to run
        self.numSteps = numsList[3]

        # Positions of QM atoms
        self.qmPos = np.genfromtxt(fin, dtype=float, usecols=(0,1,2),
                                   skip_header=1, max_rows=self.numQMAtoms)
        # Elements of QM atoms
        self.qmElmnts = np.genfromtxt(fin, dtype=str, usecols=3,
                                      skip_header=1, max_rows=self.numQMAtoms)
        self.qmElmnts = np.char.capitalize(self.qmElmnts)
        # Charges of QM atoms
        self.qmChrgs0 = np.genfromtxt(fin, dtype=float, usecols=4,
                                     skip_header=1, max_rows=self.numQMAtoms)
        # Indexes of QM atoms
        self.qmIdx = np.genfromtxt(fin, dtype=int, usecols=5,
                                   skip_header=1, max_rows=self.numQMAtoms)
        # Number of MM1 atoms which equals to number of linking atoms
        self.numMM1 = np.count_nonzero(self.qmIdx == -1)
        # Number of Real QM atoms
        self.numRealQMAtoms = self.numQMAtoms - self.numMM1

        # Positions of external point charges
        self.pntPos = np.genfromtxt(fin, dtype=float, usecols=(0,1,2),
                                    skip_header=1+self.numQMAtoms,
                                    max_rows=self.numPntChrgs)
        # Charges of external point charges
        self.pntChrgs = np.genfromtxt(fin, dtype=float, usecols=3,
                                      skip_header=1+self.numQMAtoms,
                                      max_rows=self.numPntChrgs)
        # Output external point charges
        self.outPntChrgs = self.pntChrgs
        # Indexes of external point charges
        self.pntIdx = np.genfromtxt(fin, dtype=int, usecols=4,
                                    skip_header=1+self.numQMAtoms,
                                    max_rows=self.numPntChrgs)
        # Number of virtual external point charges
        self.numVPntChrgs = np.count_nonzero(self.pntIdx == -1)
        # Number of real external point charges
        self.numRPntChrgs = self.numPntChrgs - self.numVPntChrgs

        # Numbers of MM2 atoms and virtual external point charges per MM2 atom
        if self.numVPntChrgs > 0:
            if self.pntChrgs[-1] + self.pntChrgs[-2] < 0.00001:
                self.numVPntChrgsPerMM2 = 3
            elif self.pntChrgs[-1] + self.pntChrgs[-2]*2 < 0.00001:
                self.numVPntChrgsPerMM2 = 2
            else:
                raise ValueError('Something is wrong with point charge alterations.')

            self.numMM2 = self.numVPntChrgs // self.numVPntChrgsPerMM2

        # Sort QM atoms
        self.map2sorted = np.concatenate((np.argsort(self.qmIdx[0:self.numRealQMAtoms]),
                                     np.arange(self.numRealQMAtoms, self.numQMAtoms)))
        self.map2unsorted = np.argsort(self.map2sorted)

        # Pair-wise vectors between QM and MM atoms
        self.rij = (self.qmPos[np.newaxis, :, :]
                    - self.pntPos[0:self.numRPntChrgs, np.newaxis, :])
        # Pair-wise distances between QM and MM atoms
        self.dij2 = np.sum(self.rij**2, axis=2)
        self.dij = np.sqrt(self.dij2)

    def scale_charges(self, qmSwitchingType='shift',
                      cutoff=None, swdist=None, **kwargs):
        """Scale external point charges."""
        dij_min2 = self.dij2[:, 0:self.numRealQMAtoms].min(axis=1)
        self.dij_min2 = dij_min2
        dij_min_j = self.dij2[:, 0:self.numRealQMAtoms].argmin(axis=1)
        self.dij_min_j = dij_min_j
        self.pntDist = np.sqrt(self.dij_min2)

        if qmSwitchingType.lower() == 'shift':
            if cutoff is None:
                raise ValueError("We need 'cutoff' here.")
            cutoff2 = cutoff**2
            self.pntScale = (1 - dij_min2/cutoff2)**2
            self.pntScale_deriv = 4 * (1 - dij_min2/cutoff2) / cutoff2
            self.pntScale_deriv = (self.pntScale_deriv[:,np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
        elif qmSwitchingType.lower() == 'switch':
            if cutoff is None or swdist is None:
                raise ValueError("We need 'cutoff' and 'swdist' here.")
            cutoff2 = cutoff**2
            swdist2 = swdist**2
            self.pntScale = ((dij_min2 - cutoff2)**2
                             * (cutoff2 + 2*dij_min2 - 3*swdist2)
                             / (cutoff2 - swdist2)**3
                             * (dij_min2 >= swdist2)
                             + (dij_min2 < swdist2))
            self.pntScale_deriv = (12 * (dij_min2 - swdist2)
                                   * (cutoff2 - dij_min2)
                                   / (cutoff2 - swdist2)**3)
            self.pntScale_deriv = (self.pntScale_deriv[:,np.newaxis]
                                   * (self.pntPos[0:numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
            self.pntScale_deriv *= (dij_min2 > swdist2)[:,np.newaxis]
        else:
            raise ValueError("Only 'shift' and 'switch' are supported at the moment.")

        self.pntScale = np.append(self.pntScale, np.ones(self.numVPntChrgs))
        self.pntChrgsScld = self.pntChrgs * self.pntScale
        self.outPntChrgs = self.pntChrgsScld

    def zero_pntChrgs(self):
        """Set all the external point charges to zero."""
        self.outPntChrgs = np.zeros(self.numPntChrgs)

    def get_qmparams(self, method=None, basis=None, read_first='no', read_guess=None,
                     pop=None, addparam=None):
        """Get the parameters for QM calculation."""
        if self.software.lower() == 'qchem':
            if method is not None:
                self.method = method
            else:
                raise ValueError("Please set 'method' for Q-Chem.")
            if basis is not None:
                self.basis = basis
            else:
                raise ValueError("Please set 'basis' for Q-Chem.")
            self.read_first = read_first
            if read_guess is not None:
                if read_guess.lower() == 'yes':
                    if self.stepNum == 0 and self.read_first.lower() == 'no':
                        self.read_guess = ''
                    else:
                        self.read_guess = '\nscf_guess read'
                else:
                    self.read_guess = ''
            else:
                self.read_guess = ''
            if pop is not None:
                self.pop = pop
            else:
                self.pop = 'pop_mulliken'
            if addparam is not None:
                if isinstance(addparam, list):
                    self.addparam = "".join(["%s\n" % i for i in addparam])
                else:
                    self.addparam = addparam + '\n'
            else:
                self.addparam = ''

        elif self.software.lower() == 'dftb+':
            if read_guess is not None:
                if self.stepNum == 0:
                    self.read_guess = 'No'
                else:
                    self.read_guess = read_guess
            else:
                self.read_guess = 'No'
        else:
            raise ValueError("Only 'qchem' and 'dftb+' are supported at the moment.")


    def gen_input(self, baseDir=None, **kwargs):
        """Generate input file for QM software."""

        if baseDir is not None:
            self.baseDir = baseDir
        else:
            self.baseDir = os.path.dirname(self.fin) + "/"

        if not hasattr(self, 'read_guess'):
            self.get_qmparams(**kwargs)

        qmtmplt = QMTmplt(self.software)
        qmtmplt.gen_qmtmplt()

        qmElmntsSorted = self.qmElmnts[self.map2sorted]
        qmPosSorted = self.qmPos[self.map2sorted]

        if self.software.lower() == 'qchem':
            with open(self.baseDir+"qchem.inp", "w") as f:
                f.write(qmtmplt.gen_qmtmplt().substitute(method=self.method, basis=self.basis,
                        read_guess=self.read_guess, pop=self.pop, addparam=self.addparam))
                f.write("$molecule\n")
                f.write("%d %d\n" % (self.charge, self.mult))

                for i in range(self.numQMAtoms):
                    f.write("".join(["%3s" % qmElmntsSorted[i],
                                     "%22.14e" % qmPosSorted[i,0],
                                     "%22.14e" % qmPosSorted[i,1],
                                     "%22.14e" % qmPosSorted[i,2], "\n"]))
                f.write("$end" + "\n\n")

                f.write("$external_charges\n")
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e" % self.pntPos[i,0],
                                     "%22.14e" % self.pntPos[i,1],
                                     "%22.14e" % self.pntPos[i,2],
                                     " %22.14e" % self.outPntChrgs[i], "\n"]))
                f.write("$end" + "\n")
        elif self.software.lower() == 'dftb+':
            listElmnts = np.unique(qmElmntsSorted).tolist()
            outMaxAngularMomentum = "\n    ".join([i+" = "+qmtmplt.MaxAngularMomentum[i] for i in listElmnts])
            outHubbardDerivs = "\n    ".join([i+" = "+qmtmplt.HubbardDerivs[i] for i in listElmnts])

            with open(self.baseDir+"dftb_in.hsd", "w") as f:
                f.write(qmtmplt.gen_qmtmplt().substitute(charge=self.charge,
                        numPntChrgs=self.numPntChrgs, read_guess=self.read_guess,
                        MaxAngularMomentum=outMaxAngularMomentum,
                        HubbardDerivs=outHubbardDerivs))
            with open(self.baseDir+"input_geometry.gen", "w") as f:
                f.write(str(self.numQMAtoms) + " C" + "\n")
                f.write(" ".join(listElmnts) + "\n")
                for i in range(self.numQMAtoms):
                    f.write("".join(["%6d" % (i+1),
                                    "%4d" % (listElmnts.index(qmElmntsSorted[i])+1),
                                    "%22.14e" % qmPosSorted[i,0],
                                    "%22.14e" % qmPosSorted[i,1],
                                    "%22.14e" % qmPosSorted[i,2], "\n"]))
            with open(self.baseDir+"charges.dat", 'w') as f:
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e" % self.pntPos[i,0],
                                        "%22.14e" % self.pntPos[i,1],
                                        "%22.14e" % self.pntPos[i,2],
                                        " %22.14e" % self.outPntChrgs[i], "\n"]))
        else:
            raise ValueError("Only 'qchem' and 'dftb+' are supported at the moment.")

    def run(self, **kwargs):
        """Run QM calculation."""

        if not hasattr(self, 'baseDir'):
            self.gen_input(**kwargs)

        if 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1

        cmdline = "cd " + self.baseDir + "; "

        if self.software.lower() == "qchem":
            cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

            if self.stepNum == 0 and self.read_first == 'no':
                if 'QCSCRATCH' in os.environ:
                    qcsave = os.environ['QCSCRATCH'] + "/save"
                    if os.path.isdir(qcsave):
                        shutil.rmtree(qcsave)

        elif self.software.lower() == 'dftb+':
            cmdline += "OMP_NUM_THREADS=%d dftb+ > dftb.out" % nproc

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""
        if self.software.lower() == 'qchem':
            with open(self.baseDir+"qchem.out", 'r') as f:
                for line in f:
                    line = line.strip().expandtabs()

                    if "Charge-charge energy" in line:
                        cc_energy = line.split()[-2]

                    if "Total energy" in line:
                        scf_energy = line.split()[-1]

            self.qmEnergy = float(scf_energy) - float(cc_energy)

        elif self.software.lower() == 'dftb+':
            self.qmEnergy = np.genfromtxt(self.baseDir+"results.tag",
                                          dtype=float, skip_header=1,
                                          max_rows=1)
        self.qmEnergy *= hartree2kcalmol
        return self.qmEnergy

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.qmForces = -1 * np.genfromtxt(self.baseDir+"efield.dat",
                                               dtype=float,
                                               skip_header=self.numPntChrgs)
        elif self.software.lower() == 'dftb+':
            self.qmForces = np.genfromtxt(self.baseDir+"results.tag",
                                         dtype=float, skip_header=5,
                                         max_rows=self.numQMAtoms)
        self.qmForces *= hartree2kcalmol / bohr2angstrom
        # Unsort QM atoms
        self.qmForces = self.qmForces[self.map2unsorted]
        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.pntChrgForces = (np.genfromtxt(self.baseDir+"efield.dat",
                                                dtype=float,
                                                max_rows=self.numPntChrgs)
                                  * self.outPntChrgs[:,np.newaxis])
        elif self.software.lower() == 'dftb+':
            self.pntChrgForces = np.genfromtxt(self.baseDir+"results.tag",
                                               dtype=float,
                                               skip_header=self.numQMAtoms+6,
                                               max_rows=self.numPntChrgs)
        self.pntChrgForces *= hartree2kcalmol / bohr2angstrom
        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get QM atomic charges from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.qmChrgs = np.loadtxt(self.baseDir+"charges.dat")
        elif self.software.lower() == 'dftb+':
            self.qmChrgs = np.genfromtxt(
                self.baseDir+"results.tag", dtype=float,
                skip_header=(self.numQMAtoms + self.numPntChrgs
                             + int(np.ceil(self.numQMAtoms/3.)) + 14),
                max_rows=int(np.ceil(self.numQMAtoms/3.)-1.))
            self.qmChrgs = np.append(
                self.qmChrgs.flatten(),
                np.genfromtxt(self.baseDir+"results.tag", dtype=float,
                    skip_header=(self.numQMAtoms + self.numPntChrgs
                                + int(np.ceil(self.numQMAtoms/3.))*2 + 13),
                    max_rows=1).flatten())
        # Unsort QM atoms
        self.qmChrgs = self.qmChrgs[self.map2unsorted]
        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.pntESP = np.loadtxt(self.baseDir+"esp.dat")
        elif self.software.lower() == 'dftb+':
            if not hasattr(self, 'qmChrgs'):
                self.get_qmchrgs()
            self.pntESP = (np.sum(self.qmChrgs[np.newaxis,:]
                                  / self.dij, axis=1)
                           * bohr2angstrom)
        self.pntESP *= hartree2kcalmol
        return self.pntESP

    def corr_pntchrgscale(self):
        """Correct forces due to scaling external point charges."""
        if not hasattr(self, 'pntESP'):
            self.get_pntesp()

        fCorr = self.pntESP[0:self.numRPntChrgs] * self.pntChrgs[0:self.numRPntChrgs]
        fCorr = fCorr[:,np.newaxis] * self.pntScale_deriv
        self.pntChrgForces[0:self.numRPntChrgs] += fCorr

        for i in range(self.numRealQMAtoms):
            self.qmForces[i] -= fCorr[self.dij_min_j == i].sum(axis=0)

    def corr_pbc(self):
        """Correct forces and energy due to periodic boundary conditions."""
        pntChrgsD = self.pntChrgs[0:self.numRPntChrgs] - self.outPntChrgs[0:self.numRPntChrgs]

        fCorr = -1 * ke * pntChrgsD[:,np.newaxis] * self.qmChrgs0[np.newaxis,:] / self.dij**3
        fCorr = fCorr[:,:,np.newaxis] * self.rij

        self.pntChrgForces[0:self.numRPntChrgs] += fCorr.sum(axis=1)
        self.qmForces -= fCorr.sum(axis=0)

        if hasattr(self, 'pntScale_deriv'):
            pntESP2 = ke * np.sum(self.qmChrgs0[np.newaxis,:]/self.dij, axis=1)
            fCorr = pntESP2[0:self.numRPntChrgs] * self.pntChrgs[0:self.numRPntChrgs]
            fCorr = fCorr[:,np.newaxis] * self.pntScale_deriv
            self.pntChrgForces[0:self.numRPntChrgs] -= fCorr

            for i in range(self.numRealQMAtoms):
                self.qmForces[i] += fCorr[self.dij_min_j == i].sum(axis=0)

        eCorr = ke * pntChrgsD[:,np.newaxis] * self.qmChrgs0[np.newaxis,:] / self.dij

        self.qmEnergy += eCorr.sum()

    def corr_vpntchrgs(self):
        """Correct forces due to virtual external point charges."""
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                for i in range(self.numMM2):
                    mm1Pos = (self.pntPos[self.numRPntChrgs + i*3 + 1]
                              - self.pntPos[self.numRPntChrgs + i*3]
                              * 0.94) / 0.06
                    mm2Pos = self.pntPos[self.numRPntChrgs + i*3]
                    for j in range(self.numRPntChrgs):
                        if np.abs(mm1Pos - self.pntPos[j]).sum() < 0.001:
                            mm1Idx = j
                        if np.abs(mm2Pos - self.pntPos[j]).sum() < 0.001:
                            mm2Idx = j

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i*3]
                    self.pntChrgForces[self.numRPntChrgs + i*3] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.94
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.06
                    self.pntChrgForces[self.numRPntChrgs + i*3 + 1] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * 1.06
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * -0.06
                    self.pntChrgForces[self.numRPntChrgs + i*3 + 2] = 0.
            elif self.numVPntChrgsPerMM2 == 2:
                raise ValueError("Not implemented yet.")
        else:
            pass

if __name__ == "__main__":
    import sys
    qchem = QM(sys.argv[1], 'qchem', 0, 1)
    qchem.get_qmparams(method='hf', basis='6-31g', pop='pop_mulliken')
    qchem.gen_input()
    dftb = QM(sys.argv[1], 'dftb+', 0, 1)
    dftb.get_qmparams(read_guess='No')
    dftb.gen_input()
