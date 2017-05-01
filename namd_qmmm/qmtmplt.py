from string import Template

qc_tmplt = """\
$$rem
jobtype ${jobtype}
scf_convergence 8
method ${method}
basis ${basis}
${read_guess}\
maxscf 200
qm_mm true
qmmm_charges true
igdefield 1
${addparam}\
$$end

"""

dftb_tmplt = """\
Geometry = GenFormat {
  <<< "input_geometry.gen"
}

Hamiltonian = DFTB {
  SCC = Yes
  SCCTolerance = 1e-9
  MaxSCCIterations = 200
  MaxAngularMomentum = {
    ${MaxAngularMomentum}
  }
  Charge = $charge
  SlaterKosterFiles = Type2FileNames {
    Prefix = "${skfpath}"
    Separator = "-"
    Suffix = ".skf"
    LowerCaseTypeName = No
  }
  ReadInitialCharges = ${read_guess}
  ElectricField = {
    PointCharges = {
      CoordsAndCharges [Angstrom] = DirectRead {
        Records = ${numPntChrgs}
        File = "charges.dat" }
    }
  }
  Dispersion = DftD3 {}
  DampXH = Yes
  DampXHExponent = 4.00
  ThirdOrderFull = Yes
  HubbardDerivs = {
    ${HubbardDerivs}
  }
${KPointsAndWeights}\
}

Options {
  WriteResultsTag = Yes
}

Analysis {
  WriteBandOut = No
  CalculateForces = ${calcforces}
}
${addparam}\
"""

dftbewald_tmplt = """\
  KPointsAndWeights = SupercellFolding {
    1   0   0
    0   1   0
    0   0   1
    0.0 0.0 0.0
  }
"""

HubbardDerivs = dict([('Br', '-0.0573'), ('C', '-0.1492'), ('Ca', '-0.0340'),
                      ('Cl', '-0.0697'), ('F', '-0.1623'), ('H', '-0.1857'),
                      ('I', '-0.0433'), ('K', '-0.0339'), ('Mg', '-0.02'),
                      ('N', '-0.1535'), ('Na', '-0.0454'), ('O', '-0.1575'),
                      ('P', '-0.14'), ('S', '-0.11'), ('Zn', '-0.03')])

MaxAngularMomentum = dict([('Br', 'd'), ('C', 'p'), ('Ca', 'p'),
                           ('Cl', 'd'), ('F', 'p'), ('H', 's'),
                           ('I', 'd'), ('K', 'p'), ('Mg', 'p'),
                           ('N', 'p'), ('Na', 'p'), ('O', 'p'),
                           ('P', 'd'), ('S', 'd'), ('Zn', 'd')])

orca_tmplt = """\
! ${method} ${basis} Grid4 TightSCF NOFINALGRID ${calcforces}${read_guess}${addparam}KeepDens
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs ${nproc} end
%pointcharges ${pntchrgspath}

"""

mopac_tmplt = """\
${method} XYZ T=2M 1SCF SCFCRT=1.D-7 AUX(PRECISION=9) ${calcforces}QMMM NOMM CHARGE=${charge}${addparam} THREADS=${nproc}
"""

class QMTmplt(object):
    """Input templates for QM softwares."""

    def __init__(self, qmSoftware=None, qmPBC=None):
        """Input templates for QM softwares.

        Parameters
        ----------
        qmSoftware : str
            Software to do the QM calculation

        qmPBC : bool
           If periodic boundary conditions will be used in the QM calculation

        """

        if qmSoftware is not None:
            self.qmSoftware = qmSoftware
        else:
            raise ValueError("Please choose 'qchem', 'dftb+', 'orca', 'mopac' for qmSoftware.")

        if self.qmSoftware.lower() == 'qchem':
            pass
        elif self.qmSoftware.lower() == 'dftb+':
            self.HubbardDerivs = HubbardDerivs
            self.MaxAngularMomentum = MaxAngularMomentum
        elif self.qmSoftware.lower() == 'orca':
            pass
        elif self.qmSoftware.lower() == 'mopac':
            pass
        else:
            raise ValueError("Only 'qchem', 'dftb+', 'orca', and 'mopac' are supported at the moment.")

        if qmPBC is not None:
            self.qmPBC = qmPBC
        else:
            raise ValueError("Please choose True or False for 'qmPBC'.")

    def gen_qmtmplt(self):
        """Generare input templates for QM softwares."""
        if self.qmSoftware.lower() == 'qchem':
            if self.qmPBC:
                raise ValueError("Not implemented yet.")
            else:
                return Template(qc_tmplt)
        elif self.qmSoftware.lower() == 'dftb+':
            if self.qmPBC:
                return Template(Template(dftb_tmplt).safe_substitute(KPointsAndWeights=dftbewald_tmplt))
            else:
                return Template(Template(dftb_tmplt).safe_substitute(KPointsAndWeights=''))
        elif self.qmSoftware.lower() == 'orca':
            if self.qmPBC:
                raise ValueError("Not supported.")
            else:
                return Template(orca_tmplt)
        elif self.qmSoftware.lower() == 'mopac':
            if self.qmPBC:
                raise ValueError("Not supported.")
            else:
                return Template(mopac_tmplt)


if __name__ == "__main__":
    qmtmplt = QMTmplt('qchem', False)
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              jobtype='force', method='hf', basis='6-31g',
              read_guess='scf_guess read\n',
              addparam='chelpg true\n'))
    qmtmplt = QMTmplt('dftb+', False)
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              charge=0, numPntChrgs=1000, read_guess='No',
              calcforces='Yes', addparam=''))
    qmtmplt = QMTmplt('dftb+', True)
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              charge=0, numPntChrgs=1000, read_guess='No',
              calcforces='Yes', addparam=''))
    qmtmplt = QMTmplt('orca', False)
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              method='HF', basis='6-31G', calcforces='EnGrad ',
              read_guess='NoAutoStart ',
              addparam='CHELPG ', nproc='8'))
    qmtmplt = QMTmplt('mopac', False)
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              method='PM7', calcforces='GRAD ',
              charge=0, addparam=' ESP', nproc='8'))
