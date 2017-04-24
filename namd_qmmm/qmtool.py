import os
import shutil
import subprocess as sp
import numpy as np
from .qmtmplt import QMTmplt

HARTREE2KCALMOL = 6.275094737775374e+02
BOHR2ANGSTROM = 5.2917721067e-01
KE = HARTREE2KCALMOL * BOHR2ANGSTROM


class QM(object):
    def __init__(self, fin, software=None, charge=None, mult=None, pbc=None):
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
            raise ValueError("Please choose 'qchem', 'dftb+', or 'orca' for qmSoftware.")
        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1
        if pbc is not None:
            self.pbc = pbc
        else:
            raise ValueError("Please set 'pbc' for QM calculation.")

        # Load system information
        sysList = np.genfromtxt(fin, dtype=int, max_rows=1, unpack=True)
        # Number of QM atoms including linking atoms
        self.numQMAtoms = sysList[0]
        # Number of external external point charges including virtual particles
        self.numPntChrgs = sysList[1]
        # Number of total atoms in the whole system
        self.numAtoms = sysList[2]
        # Number of current step
        self.stepNum = sysList[3]
        # Number of total steps to run
        self.numSteps = sysList[4]

        # Load QM information
        qmList = np.genfromtxt(fin, dtype=None, skip_header=1,
                               max_rows=self.numQMAtoms)
        # Positions of QM atoms
        self.qmPos = np.column_stack((qmList['f0'],
                                      qmList['f1'],
                                      qmList['f2']))
        # Elements of QM atoms
        self.qmElmnts = np.char.capitalize(np.core.defchararray.decode(qmList['f3']))
        # Charges of QM atoms
        self.qmChrgs0 = qmList['f4']
        # Indexes of QM atoms
        self.qmIdx = qmList['f5']

        # Number of MM1 atoms which equals to number of linking atoms
        self.numMM1 = np.count_nonzero(self.qmIdx == -1)
        # Number of Real QM atoms
        self.numRealQMAtoms = self.numQMAtoms - self.numMM1

        # Load external point charge information
        pntList = np.genfromtxt(fin, dtype=None, skip_header=1+self.numQMAtoms,
                                max_rows=self.numPntChrgs)
        # Positions of external point charges
        self.pntPos = np.column_stack((pntList['f0'],
                                       pntList['f1'],
                                       pntList['f2']))
        # Charges of external point charges
        self.pntChrgs = pntList['f3']
        # Indexes of external point charges
        self.pntIdx = pntList['f4']
        # Indexes of QM atoms MM1 atoms bonded to
        self.pntBondedToIdx = pntList['f5']

        # Local indexes of MM1 and QM host atoms
        if self.numMM1 > 0:
            self.mm1LocalIdx = np.where(self.pntBondedToIdx != -1)[0]
            self.qmHostLocalIdx = self.pntBondedToIdx[self.mm1LocalIdx]
        # Number of virtual external point charges
        self.numVPntChrgs = np.count_nonzero(self.pntIdx == -1)
        # Number of real external point charges
        self.numRPntChrgs = self.numPntChrgs - self.numVPntChrgs

        # Numbers of MM2 atoms and virtual external point charges per MM2 atom
        if self.numVPntChrgs > 0:
            if self.pntChrgs[-1] + self.pntChrgs[-2] < 0.00001:
                self.numVPntChrgsPerMM2 = 3
            elif self.pntChrgs[-1] + self.pntChrgs[-2] * 2 < 0.00001:
                self.numVPntChrgsPerMM2 = 2
            else:
                raise ValueError('Something is wrong with point charge alterations.')

            self.numMM2 = self.numVPntChrgs // self.numVPntChrgsPerMM2

        # Local indexes of MM1 and MM2 atoms the virtual point charges belong to
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                mm1VPos = np.zeros((self.numMM2, 3), dtype=float)
                mm2VPos = np.zeros((self.numMM2, 3), dtype=float)
                for i in range(self.numMM2):
                    mm1VPos[i] = (self.pntPos[self.numRPntChrgs + i*3 + 1]
                                  - self.pntPos[self.numRPntChrgs + i*3]
                                  * 0.94) / 0.06
                    mm2VPos[i] = self.pntPos[self.numRPntChrgs + i*3]

                self.mm1VIdx = np.zeros(self.numMM2, dtype=int)
                self.mm2VIdx = np.zeros(self.numMM2, dtype=int)
                for i in range(self.numMM2):
                    for j in range(self.numMM1):
                        if np.abs(mm1VPos[i] - self.pntPos[self.mm1LocalIdx[j]]).sum() < 0.001:
                            self.mm1VIdx[i] = self.mm1LocalIdx[j]
                            break
                for i in range(self.numMM2):
                    for j in range(self.numRPntChrgs):
                        if np.abs(mm2VPos[i] - self.pntPos[j]).sum() < 0.001:
                            self.mm2VIdx[i] = j
                            break
                self.mm2LocalIdx = []
                for i in range(self.numMM1):
                    self.mm2LocalIdx.append(self.mm2VIdx[self.mm1VIdx == self.mm1LocalIdx[i]])
            elif self.numVPntChrgsPerMM2 == 2:
                raise ValueError("Not implemented yet.")

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

        # Load unit cell information
        if self.pbc.lower() == 'yes':
            if self.numAtoms != self.numRealQMAtoms+self.numRPntChrgs:
                raise ValueError("Unit cell is not complete.")

            cellList = np.genfromtxt(fin, dtype=None, skip_header=1+self.numQMAtoms+self.numPntChrgs,
                                     max_rows=4)
            self.cellOrigin = cellList[0]
            self.cellBasisVector1 = cellList[1]
            self.cellBasisVector2 = cellList[2]
            self.cellBasisVector3 = cellList[3]

    def scale_charges(self, qmSwitchingType=None,
                      qmCutoff=None, qmSwdist=None, **kwargs):
        """Scale external point charges."""
        dij_min2 = self.dij2[:, 0:self.numRealQMAtoms].min(axis=1)
        self.dij_min2 = dij_min2
        dij_min_j = self.dij2[:, 0:self.numRealQMAtoms].argmin(axis=1)
        self.dij_min_j = dij_min_j
        self.pntDist = np.sqrt(self.dij_min2)

        if qmSwitchingType.lower() == 'shift':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            qmCutoff2 = qmCutoff**2
            self.pntScale = (1 - dij_min2/qmCutoff2)**2
            self.pntScale_deriv = 4 * (1 - dij_min2/qmCutoff2) / qmCutoff2
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
        elif qmSwitchingType.lower() == 'switch':
            if qmCutoff is None or qmSwdist is None:
                raise ValueError("We need qmCutoff and qmSwdist here.")
            if qmCutoff <= qmSwdist:
                raise ValueError("qmCutoff should be greater than qmSwdist.")
            qmCutoff2 = qmCutoff**2
            qmSwdist2 = qmSwdist**2
            self.pntScale = ((dij_min2 - qmCutoff2)**2
                             * (qmCutoff2 + 2*dij_min2 - 3*qmSwdist2)
                             / (qmCutoff2 - qmSwdist2)**3
                             * (dij_min2 >= qmSwdist2)
                             + (dij_min2 < qmSwdist2))
            self.pntScale_deriv = (12 * (dij_min2 - qmSwdist2)
                                   * (qmCutoff2 - dij_min2)
                                   / (qmCutoff2 - qmSwdist2)**3)
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
            self.pntScale_deriv *= (dij_min2 > qmSwdist2)[:, np.newaxis]
        elif qmSwitchingType.lower() == 'lrec':
            if qmCutoff is None:
                raise ValueError("We need qmCutoff here.")
            scale = 1 - self.pntDist / qmCutoff
            self.pntScale = 1 - (2*scale**3 - 3*scale**2 + 1)**2
            self.pntScale_deriv = 12 * scale * (2*scale**3 - 3*scale**2 + 1) / qmCutoff**2
            self.pntScale_deriv = (self.pntScale_deriv[:, np.newaxis]
                                   * (self.pntPos[0:self.numRPntChrgs]
                                   - self.qmPos[dij_min_j]))
        else:
            raise ValueError("Only 'shift', 'switch', and 'lrec' are supported at the moment.")

        # Just to be safe
        self.pntScale *= (self.pntDist < qmCutoff)
        self.pntScale_deriv *= (self.pntDist < qmCutoff)[:, np.newaxis]

        self.pntScale = np.append(self.pntScale, np.ones(self.numVPntChrgs))
        self.pntChrgsScld = self.pntChrgs * self.pntScale

    def get_qmparams(self, method=None, basis=None, read_first='no',
                     read_guess=None, calc_forces=None, pop=None, addparam=None):
        """Get the parameters for QM calculation."""
        if self.software.lower() == 'qchem':
            if method is not None:
                self.method = method
            else:
                raise ValueError("Please set method for Q-Chem.")

            if basis is not None:
                self.basis = basis
            else:
                raise ValueError("Please set basis for Q-Chem.")

            if pop is not None:
                self.pop = pop
            elif not hasattr(self, 'pop'):
                self.pop = 'mulliken'

            if addparam is not None:
                if isinstance(addparam, list):
                    self.addparam = "".join(["%s\n" % i for i in addparam])
                else:
                    self.addparam = addparam + '\n'
            else:
                self.addparam = ''

        elif self.software.lower() == 'dftb+':
            pass

        elif self.software.lower() == 'orca':
            if method is not None:
                self.method = method
            else:
                raise ValueError("Please set method for ORCA.")

            if basis is not None:
                self.basis = basis
            else:
                raise ValueError("Please set basis for ORCA.")

            if pop is not None:
                self.pop = pop
            elif not hasattr(self, 'pop'):
                self.pop = 'mulliken'

            if addparam is not None:
                if isinstance(addparam, list):
                    self.addparam = "".join(["%s " % i for i in addparam])
                else:
                    self.addparam = addparam + " "
            else:
                self.addparam = ''

        else:
            raise ValueError("Only 'qchem', 'dftb+', and 'orca' are supported at the moment.")

        if calc_forces is not None:
            self.calc_forces = calc_forces
        elif not hasattr(self, 'calc_forces'):
            self.calc_forces = 'yes'

        self.read_first = read_first

        if read_guess is not None:
            if self.stepNum == 0 and self.read_first.lower() == 'no':
                self.read_guess = 'no'
            else:
                self.read_guess = read_guess
        else:
            self.read_guess = 'no'

    def get_nproc(self):
        """Get the number of processes for QM calculation."""
        if 'OMP_NUM_THREADS' in os.environ:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        elif 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1
        return nproc

    def gen_input(self, baseDir=None, **kwargs):
        """Generate input file for QM software."""

        if baseDir is not None:
            self.baseDir = baseDir
        else:
            self.baseDir = os.path.dirname(self.fin) + "/"

        if not hasattr(self, 'read_guess'):
            self.get_qmparams(**kwargs)

        qmtmplt = QMTmplt(self.software, self.pbc)

        qmElmntsSorted = self.qmElmnts[self.map2sorted]
        qmPosSorted = self.qmPos[self.map2sorted]
        qmIdxSorted = self.qmIdx[self.map2sorted]

        if self.software.lower() == 'qchem' and self.pbc.lower() == 'no':

            if self.calc_forces.lower() == 'yes':
                jobtype = 'force'
            elif self.calc_forces.lower() == 'no':
                jobtype = 'sp'

            if self.read_guess.lower() == 'yes':
                read_guess = '\nscf_guess read'
            elif self.read_guess.lower() == 'no':
                read_guess = ''

            if self.pop == 'mulliken':
                pop = ''
            elif self.pop == 'esp':
                pop = '\nesp_charges true'
            elif self.pop == 'chelpg':
                pop = '\nchelpg true'

            with open(self.baseDir+"qchem.inp", "w") as f:
                f.write(qmtmplt.gen_qmtmplt().substitute(jobtype=jobtype,
                        method=self.method, basis=self.basis,
                        read_guess=read_guess, pop=pop,
                        addparam=self.addparam))
                f.write("$molecule\n")
                f.write("%d %d\n" % (self.charge, self.mult))

                for i in range(self.numQMAtoms):
                    f.write("".join(["%3s" % qmElmntsSorted[i],
                                     "%22.14e" % qmPosSorted[i, 0],
                                     "%22.14e" % qmPosSorted[i, 1],
                                     "%22.14e" % qmPosSorted[i, 2], "\n"]))
                f.write("$end" + "\n\n")

                f.write("$external_charges\n")
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e" % self.pntPos[i, 0],
                                     "%22.14e" % self.pntPos[i, 1],
                                     "%22.14e" % self.pntPos[i, 2],
                                     " %22.14e" % self.pntChrgs4QM[i], "\n"]))
                f.write("$end" + "\n")

        elif self.software.lower() == 'dftb+':

            listElmnts = np.unique(qmElmntsSorted).tolist()
            outMaxAngularMomentum = "\n    ".join([i+" = "+qmtmplt.MaxAngularMomentum[i] for i in listElmnts])
            outHubbardDerivs = "\n    ".join([i+" = "+qmtmplt.HubbardDerivs[i] for i in listElmnts])

            if self.calc_forces.lower() == 'yes':
                calcforces = 'Yes'
            elif self.calc_forces.lower() == 'no':
                calcforces = 'No'

            if self.read_guess.lower() == 'yes':
                read_guess = 'Yes'
            elif self.read_guess.lower() == 'no':
                read_guess = 'No'

            with open(self.baseDir+"dftb_in.hsd", 'w') as f:
                f.write(qmtmplt.gen_qmtmplt().substitute(charge=self.charge,
                        numPntChrgs=self.numPntChrgs, read_guess=read_guess,
                        calcforces=calcforces,
                        MaxAngularMomentum=outMaxAngularMomentum,
                        HubbardDerivs=outHubbardDerivs))
            with open(self.baseDir+"input_geometry.gen", 'w') as f:
                if self.pbc.lower() == 'no':
                    f.write(str(self.numQMAtoms) + " C" + "\n")
                elif self.pbc.lower() == 'yes':
                    f.write(str(self.numQMAtoms) + " S" + "\n")
                f.write(" ".join(listElmnts) + "\n")
                for i in range(self.numQMAtoms):
                    f.write("".join(["%6d" % (i+1),
                                     "%4d" % (listElmnts.index(qmElmntsSorted[i])+1),
                                     "%22.14e" % qmPosSorted[i, 0],
                                     "%22.14e" % qmPosSorted[i, 1],
                                     "%22.14e" % qmPosSorted[i, 2], "\n"]))
                if self.pbc.lower() == 'yes':
                    f.write("".join(["%22.14e" % i for i in self.cellOrigin]) + "\n")
                    f.write("".join(["%22.14e" % i for i in self.cellBasisVector1]) + "\n")
                    f.write("".join(["%22.14e" % i for i in self.cellBasisVector2]) + "\n")
                    f.write("".join(["%22.14e" % i for i in self.cellBasisVector3]) + "\n")

            with open(self.baseDir+"charges.dat", 'w') as f:
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e" % self.pntPos[i, 0],
                                     "%22.14e" % self.pntPos[i, 1],
                                     "%22.14e" % self.pntPos[i, 2],
                                     " %22.14e" % self.pntChrgs4QM[i], "\n"]))

        elif self.software.lower() == 'orca':

            if self.calc_forces.lower() == 'yes':
                calcforces = 'EnGrad '
            elif self.calc_forces.lower() == 'no':
                calcforces = ''

            if self.read_guess.lower() == 'yes':
                read_guess = ''
            elif self.read_guess.lower() == 'no':
                read_guess = 'NoAutoStart '

            if self.pop == 'mulliken':
                pop = ''
            elif self.pop == 'chelpg':
                pop = 'CHELPG '

            nproc = get_nproc()

            with open(self.baseDir + "orca.inp", 'w') as f:
                f.write(qmtmplt.gen_qmtmplt().substitute(
                        method=self.method, basis=self.basis,
                        calcforces=calcforces, read_guess=read_guess,
                        pop=pop, addparam=self.addparam, nproc=nproc,
                        pntchrgspath="\"orca.pntchrg\""))
                f.write("%coords\n")
                f.write("  CTyp xyz\n")
                f.write("  Charge %d\n" % self.charge)
                f.write("  Mult %d\n" % self.mult)
                f.write("  Units Angs\n")
                f.write("  coords\n")

                for i in range(self.numQMAtoms):
                    f.write(" ".join(["%6s" % qmElmntsSorted[i],
                                     "%22.14e" % qmPosSorted[i, 0],
                                     "%22.14e" % qmPosSorted[i, 1],
                                     "%22.14e" % qmPosSorted[i, 2], "\n"]))
                f.write("  end\n")
                f.write("end\n")

            with open(self.baseDir + "orca.pntchrg", 'w') as f:
                f.write("%d\n" % self.numPntChrgs)
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e " % self.pntChrgs4QM[i],
                                     "%22.14e" % self.pntPos[i, 0],
                                     "%22.14e" % self.pntPos[i, 1],
                                     "%22.14e" % self.pntPos[i, 2], "\n"]))

            with open(self.baseDir + "orca.pntvpot.xyz", 'w') as f:
                f.write("%d\n" % self.numPntChrgs)
                for i in range(self.numPntChrgs):
                    f.write("".join(["%22.14e" % (self.pntPos[i, 0] / BOHR2ANGSTROM),
                                     "%22.14e" % (self.pntPos[i, 1] / BOHR2ANGSTROM),
                                     "%22.14e" % (self.pntPos[i, 2] / BOHR2ANGSTROM), "\n"]))

        elif self.software.lower() == 'qchem' and self.pbc.lower() == 'yes':
            raise ValueError("Not implemented yet.")
        else:
            raise ValueError("Only 'qchem' and 'dftb+' are supported at the moment.")

    def run(self, **kwargs):
        """Run QM calculation."""

        if not hasattr(self, 'baseDir'):
            self.gen_input(**kwargs)

        nproc = get_nproc()

        cmdline = "cd " + self.baseDir + "; "

        if self.software.lower() == 'qchem':
            cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

            if self.stepNum == 0 and self.read_first.lower() == 'no':
                if 'QCSCRATCH' in os.environ:
                    qcsave = os.environ['QCSCRATCH'] + "/save"
                    if os.path.isdir(qcsave):
                        shutil.rmtree(qcsave)

        elif self.software.lower() == 'dftb+':
            cmdline += "export OMP_NUM_THREADS=%d; dftb+ > dftb.out" % nproc

        if self.software.lower() == 'orca':
            cmdline += "orca orca.inp > orca.out; "
            cmdline += "orca_vpot orca.gbw orca.scfp orca.pntvpot.xyz orca.pntvpot.out >> orca.out"

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    def get_qmenergy(self):
        """Get QM energy from output of QM calculation."""
        if self.software.lower() == 'qchem':
            with open(self.baseDir + "qchem.out", 'r') as f:
                for line in f:
                    line = line.strip().expandtabs()

                    if "Charge-charge energy" in line:
                        cc_energy = line.split()[-2]

                    if "Total energy" in line:
                        scf_energy = line.split()[-1]
                        break

            self.qmEnergy = float(scf_energy) - float(cc_energy)

        elif self.software.lower() == 'dftb+':
            self.qmEnergy = np.genfromtxt(self.baseDir + "results.tag",
                                          dtype=float, skip_header=1,
                                          max_rows=1)

        elif self.software.lower() == 'orca':
            with open(self.baseDir + "orca.out", 'r') as f:
                for line in f:
                    line = line.strip().expandtabs()

                    if "FINAL SINGLE POINT ENERGY" in line:
                        self.qmEnergy = float(line.split()[-1])
                        break

        self.qmEnergy *= HARTREE2KCALMOL
        return self.qmEnergy

    def get_qmforces(self):
        """Get QM forces from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.qmForces = -1 * np.genfromtxt(self.baseDir + "efield.dat",
                                               dtype=float,
                                               skip_header=self.numPntChrgs)
        elif self.software.lower() == 'dftb+':
            self.qmForces = np.genfromtxt(self.baseDir + "results.tag",
                                          dtype=float, skip_header=5,
                                          max_rows=self.numQMAtoms)
        elif self.software.lower() == 'orca':
            self.qmForces = -1 * np.genfromtxt(self.baseDir +"orca.engrad",
                                               dtype=float, skip_header=11,
                                               max_rows=self.numQMAtoms*3).reshape((self.numQMAtoms, 3))
        self.qmForces *= HARTREE2KCALMOL / BOHR2ANGSTROM
        # Unsort QM atoms
        self.qmForces = self.qmForces[self.map2unsorted]
        return self.qmForces

    def get_pntchrgforces(self):
        """Get external point charge forces from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.pntChrgForces = (np.genfromtxt(self.baseDir + "efield.dat",
                                                dtype=float,
                                                max_rows=self.numPntChrgs)
                                  * self.pntChrgs4QM[:, np.newaxis])
        elif self.software.lower() == 'dftb+':
            self.pntChrgForces = np.genfromtxt(self.baseDir + "results.tag",
                                               dtype=float,
                                               skip_header=self.numQMAtoms+6,
                                               max_rows=self.numPntChrgs)
        elif self.software.lower() == 'orca':
            self.pntChrgForces = -1 * np.genfromtxt(self.baseDir + "orca.pcgrad",
                                                    dtype=float,
                                                    skip_header=1,
                                                    max_rows=self.numPntChrgs)
        self.pntChrgForces *= HARTREE2KCALMOL / BOHR2ANGSTROM
        return self.pntChrgForces

    def get_qmchrgs(self):
        """Get QM atomic charges from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.qmChrgs = np.loadtxt(self.baseDir + "charges.dat")
        elif self.software.lower() == 'dftb+':
            if self.numQMAtoms > 3:
                self.qmChrgs = np.genfromtxt(
                    self.baseDir + "results.tag", dtype=float,
                    skip_header=(self.numQMAtoms + self.numPntChrgs
                                 + int(np.ceil(self.numQMAtoms/3.)) + 14),
                    max_rows=int(np.ceil(self.numQMAtoms/3.)-1.))
            else:
                self.qmChrgs = np.array([])
            self.qmChrgs = np.append(
                self.qmChrgs.flatten(),
                np.genfromtxt(self.baseDir + "results.tag", dtype=float,
                    skip_header=(self.numQMAtoms + self.numPntChrgs
                                 + int(np.ceil(self.numQMAtoms/3.))*2 + 13),
                    max_rows=1).flatten())
        elif self.software.lower() == 'orca':
            if self.pop == 'mulliken':
                beg = "MULLIKEN ATOMIC CHARGES"
            elif self.pop == 'chelpg':
                beg = "CHELPG Charges"
            with open(self.baseDir + "orca.out", 'r') as f:
                for line in f:
                    if beg in line:
                        charges = []
                        line = next(f)
                        for i in range(self.numQMAtoms):
                            line = next(f)
                            charges.append(float(line.split()[3]))
                        break
            self.qmChrgs = np.array(charges)

        # Unsort QM atoms
        self.qmChrgs = self.qmChrgs[self.map2unsorted]
        return self.qmChrgs

    def get_pntesp(self):
        """Get ESP at external point charges from output of QM calculation."""
        if self.software.lower() == 'qchem':
            self.pntESP = np.loadtxt(self.baseDir + "esp.dat")
        elif self.software.lower() == 'dftb+':
            if not hasattr(self, 'qmChrgs'):
                self.get_qmchrgs()
            self.pntESP = (np.sum(self.qmChrgs[np.newaxis, :]
                                  / self.dij, axis=1)
                           * BOHR2ANGSTROM)
        if self.software.lower() == 'orca':
            self.pntESP = np.genfromtxt(self.baseDir + "orca.pntvpot.out",
                                        dtype=float,
                                        skip_header=1,
                                        usecols=3,
                                        max_rows=self.numPntChrgs)
        self.pntESP *= HARTREE2KCALMOL
        return self.pntESP

    def corr_pntchrgscale(self):
        """Correct forces due to scaling external point charges."""
        if not hasattr(self, 'pntESP'):
            self.get_pntesp()

        fCorr = self.pntESP[0:self.numRPntChrgs] * self.pntChrgs[0:self.numRPntChrgs]
        fCorr = fCorr[:, np.newaxis] * self.pntScale_deriv
        self.pntChrgForces[0:self.numRPntChrgs] += fCorr

        for i in range(self.numRealQMAtoms):
            self.qmForces[i] -= fCorr[self.dij_min_j == i].sum(axis=0)

    def corr_qmpntchrgs(self):
        """Correct forces and energy due to using partial charges for QM atoms."""
        pntChrgsD = self.pntChrgs4MM[0:self.numRPntChrgs] - self.pntChrgs4QM[0:self.numRPntChrgs]

        fCorr = -1 * KE * pntChrgsD[:, np.newaxis] * self.qmChrgs4MM[np.newaxis, :] / self.dij**3
        fCorr = fCorr[:, :, np.newaxis] * self.rij

        if self.numVPntChrgs > 0:
            for i in range(self.numMM1):
                fCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0

        self.pntChrgForces[0:self.numRPntChrgs] += fCorr.sum(axis=1)
        self.qmForces -= fCorr.sum(axis=0)

        if hasattr(self, 'pntChrgsScld') and self.pntChrgs4QM is not self.pntChrgs4MM:
            fCorr = KE * self.pntChrgs[0:self.numRPntChrgs, np.newaxis] * self.qmChrgs4MM[np.newaxis, :] / self.dij
            if self.numVPntChrgs > 0:
                for i in range(self.numMM1):
                    fCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0
            fCorr = np.sum(fCorr, axis=1)
            fCorr = fCorr[:, np.newaxis] * self.pntScale_deriv

            if self.pntChrgs4MM is self.pntChrgsScld:
                fCorr *= -1

            self.pntChrgForces[0:self.numRPntChrgs] -= fCorr
            for i in range(self.numRealQMAtoms):
                self.qmForces[i] += fCorr[self.dij_min_j == i].sum(axis=0)

        eCorr = KE * pntChrgsD[:, np.newaxis] * self.qmChrgs4MM[np.newaxis, :] / self.dij

        if self.numVPntChrgs > 0:
            for i in range(self.numMM1):
                eCorr[self.mm2LocalIdx[i], self.qmHostLocalIdx[i]] = 0.0

        self.qmEnergy += eCorr.sum()

    def corr_vpntchrgs(self):
        """Correct forces due to virtual external point charges."""
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                for i in range(self.numMM2):
                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3]

                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.94
                    self.pntChrgForces[self.mm1VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 1] * 0.06

                    self.pntChrgForces[self.mm2VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * 1.06
                    self.pntChrgForces[self.mm1VIdx[i]] += self.pntChrgForces[self.numRPntChrgs + i*3 + 2] * -0.06

                self.pntChrgForces[self.numRPntChrgs:] = 0.

            elif self.numVPntChrgsPerMM2 == 2:
                raise ValueError("Not implemented yet.")
        else:
            pass

    def corr_vpntchrgs_old(self):
        """Correct forces due to virtual external point charges (deprecated)."""
        if self.numVPntChrgs > 0:
            if self.numVPntChrgsPerMM2 == 3:
                for i in range(self.numMM2):
                    mm1Pos = (self.pntPos[self.numRPntChrgs + i * 3 + 1]
                              - self.pntPos[self.numRPntChrgs + i * 3]
                              * 0.94) / 0.06
                    mm2Pos = self.pntPos[self.numRPntChrgs + i * 3]
                    for j in range(self.numRPntChrgs):
                        if np.abs(mm1Pos - self.pntPos[j]).sum() < 0.001:
                            mm1Idx = j
                        if np.abs(mm2Pos - self.pntPos[j]).sum() < 0.001:
                            mm2Idx = j

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3]
                    self.pntChrgForces[self.numRPntChrgs + i * 3] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] * 0.94
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] * 0.06
                    self.pntChrgForces[self.numRPntChrgs + i * 3 + 1] = 0.

                    self.pntChrgForces[mm2Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] * 1.06
                    self.pntChrgForces[mm1Idx] += self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] * -0.06
                    self.pntChrgForces[self.numRPntChrgs + i * 3 + 2] = 0.
            elif self.numVPntChrgsPerMM2 == 2:
                raise ValueError("Not implemented yet.")
        else:
            pass


if __name__ == "__main__":
    import sys
    qchem = QM(sys.argv[1], 'qchem', 0, 1)
    qchem.get_qmparams(method='hf', basis='6-31g', pop='mulliken')
    qchem.gen_input()
    dftb = QM(sys.argv[1], 'dftb+', 0, 1)
    dftb.get_qmparams(read_guess='No')
    dftb.gen_input()
