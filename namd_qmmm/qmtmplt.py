#!/usr/bin/env python3
from string import Template

qc_tmplt="""\
$$rem
jobtype force
method $method
basis $basis
scf_guess $scf_guess
maxscf 200
qm_mm true
qmmm_charges true
$pop true
igdefield 1
$$end

"""

dftb_tmplt="""\
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
  ReadInitialCharges = $initial_scc
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
}

Options {
  WriteResultsTag = Yes
}

Analysis {
  WriteBandOut = No
  CalculateForces = Yes
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

class QMTmplt(object):
    """Class for input templates for QM softwares."""
    def __init__(self, qmSoftware=None):
        if qmSoftware is None:
            raise ValueError("Please choose 'qmSoftware' from 'qchem' and 'dftb+'.")
        else:
            self.qmSoftware = qmSoftware

        if self.qmSoftware.lower() == 'qchem':
            pass
        elif self.qmSoftware.lower() == 'dftb+':
            self.HubbardDerivs = HubbardDerivs
            self.MaxAngularMomentum = MaxAngularMomentum
        else:
            raise ValueError("Only 'qchem' and 'dftb+' are supported at the moment.")

    def gen_qmtmplt(self):
        """Generare input templates for QM softwares."""
        if self.qmSoftware.lower() == 'qchem':
            return Template(qc_tmplt)
        elif self.qmSoftware.lower() == 'dftb+':
            return Template(dftb_tmplt)

if __name__ == "__main__":
    qchem = QMTmplt('qchem')
    print(qchem.gen_qmtmplt().substitute(
              method='hf', basis='6-31g',
              scf_guess='sad', pop='pop_mulliken'))
    dftb = QMTmplt('dftb+')
    print(dftb.gen_qmtmplt().safe_substitute(
              charge=0, numPntChrgs=1000, initial_scc='No'))
