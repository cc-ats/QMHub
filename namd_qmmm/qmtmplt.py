from string import Template

qc_tmplt = """\
$$rem
jobtype $jobtype
scf_convergence 8
method $method
basis $basis\
$read_guess
maxscf 200
qm_mm true
qmmm_charges true\
$pop
igdefield 1
$addparam\
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
    $MaxAngularMomentum
  }
  Charge = $charge
  SlaterKosterFiles = Type2FileNames {
    Prefix = "/home/panxl/dftb/3ob-3-1/"
    Separator = "-"
    Suffix = ".skf"
    LowerCaseTypeName = No
  }
  ReadInitialCharges = $read_guess
  ElectricField = {
    PointCharges = {
      CoordsAndCharges [Angstrom] = DirectRead {
        Records = $numPntChrgs
        File = "charges.dat" }
    }
  }
  Dispersion = DftD3 {}
  DampXH = Yes
  DampXHExponent = 4.00
  ThirdOrderFull = Yes
  HubbardDerivs = {
    $HubbardDerivs
  }
$KPointsAndWeights\
}

Options {
  WriteResultsTag = Yes
}

Analysis {
  WriteBandOut = No
  CalculateForces = $calcforces
}
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
! $method $basis Grid4 TightSCF NOFINALGRID ${calcforces}${read_guess}${pop}${addparam}KeepDens
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs $nproc end
%pointcharges $pntchrgspath

"""


class QMTmplt(object):
    """Class for input templates for QM softwares."""
    def __init__(self, qmSoftware=None, qmPBC=None):
        if qmSoftware is not None:
            self.qmSoftware = qmSoftware
        else:
            raise ValueError("Please choose 'qchem', 'dftb+', or 'orca' for qmSoftware.")

        if self.qmSoftware.lower() == 'qchem':
            pass
        elif self.qmSoftware.lower() == 'dftb+':
            self.HubbardDerivs = HubbardDerivs
            self.MaxAngularMomentum = MaxAngularMomentum
        elif self.qmSoftware.lower() == 'orca':
            pass
        else:
            raise ValueError("Only 'qchem', 'dftb+', and 'orca' are supported at the moment.")

        if qmPBC is not None:
            self.qmPBC = qmPBC
        else:
            raise ValueError("Please choose 'yes' or 'no' for 'qmPBC'.")

    def gen_qmtmplt(self):
        """Generare input templates for QM softwares."""
        if self.qmSoftware.lower() == 'qchem':
            if self.qmPBC.lower() == 'no':
                return Template(qc_tmplt)
            if self.qmPBC.lower() == 'yes':
                raise ValueError("Not implemented yet.")
        elif self.qmSoftware.lower() == 'dftb+':
            if self.qmPBC.lower() == 'no':
                return Template(Template(dftb_tmplt).safe_substitute(KPointsAndWeights=''))
            if self.qmPBC.lower() == 'yes':
                return Template(Template(dftb_tmplt).safe_substitute(KPointsAndWeights=dftbewald_tmplt))
        if self.qmSoftware.lower() == 'orca':
            if self.qmPBC.lower() == 'no':
                return Template(orca_tmplt)
            if self.qmPBC.lower() == 'yes':
                raise ValueError("Not supported.")


if __name__ == "__main__":
    qmtmplt = QMTmplt('qchem', 'no')
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              jobtype='force', method='hf', basis='6-31g',
              read_guess='\nscf_guess read', pop='\nchelpg true',
              addparam='esp_charges true\n'))
    qmtmplt = QMTmplt('dftb+', 'no')
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              charge=0, numPntChrgs=1000, read_guess='No',
              calcforces='Yes'))
    qmtmplt = QMTmplt('dftb+', 'yes')
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              charge=0, numPntChrgs=1000, read_guess='No',
              calcforces='Yes'))
    qmtmplt = QMTmplt('orca', 'no')
    print(qmtmplt.gen_qmtmplt().safe_substitute(
              method='HF', basis='6-31G', calcforces='EnGrad ',
              read_guess='NoAutoStart ', pop='CHELPG ',
              addparam='MAYER ', nproc='8'))
