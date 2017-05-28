from string import Template

qc_tmpl = """\
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

dftb_tmpl = """\
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
        Records = ${n_mm_atoms}
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
  CalculateForces = ${calc_forces}
}
${addparam}\
"""

KPointsAndWeights = """\
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

orca_tmpl = """\
! ${method} ${basis} Grid4 TightSCF NOFINALGRID ${calc_forces}${read_guess}${addparam}KeepDens
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs ${nproc} end
%pointcharges ${pntchrgspath}

"""

mopac_tmpl = """\
${method} XYZ T=2M 1SCF SCFCRT=1.D-7 AUX(PRECISION=9) ${calc_forces}QMMM NOMM CHARGE=${charge}${addparam} THREADS=${nproc}
"""

class QMTmpl(object):
    """Input templates for QM softwares."""

    def __init__(self, qmSoftware=None):
        """Input templates for QM softwares.

        Parameters
        ----------
        qmSoftware : str
            Software to do the QM calculation

        """

        if qmSoftware is not None:
            self.qmSoftware = qmSoftware
        else:
            raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac' for qmSoftware.")

        if self.qmSoftware.lower() == 'q-chem':
            pass
        elif self.qmSoftware.lower() == 'dftb+':
            self.KPointsAndWeights = KPointsAndWeights
            self.HubbardDerivs = HubbardDerivs
            self.MaxAngularMomentum = MaxAngularMomentum
        elif self.qmSoftware.lower() == 'orca':
            pass
        elif self.qmSoftware.lower() == 'mopac':
            pass
        else:
            raise ValueError("Only 'q-chem', 'dftb+', 'orca', and 'mopac' are supported at the moment.")

    def gen_qmtmpl(self):
        """Generare input templates for QM softwares."""
        if self.qmSoftware.lower() == 'q-chem':
            return Template(qc_tmpl)
        elif self.qmSoftware.lower() == 'dftb+':
            return Template(dftb_tmpl)
        elif self.qmSoftware.lower() == 'orca':
            return Template(orca_tmpl)
        elif self.qmSoftware.lower() == 'mopac':
            return Template(mopac_tmpl)


if __name__ == "__main__":
    qmtmpl = QMTmpl('q-chem')
    print(qmtmpl.gen_qmtmpl().safe_substitute(
              jobtype='force', method='hf', basis='6-31g',
              read_guess='scf_guess read\n',
              addparam='chelpg true\n'))
    qmtmpl = QMTmpl('dftb+')
    print(qmtmpl.gen_qmtmpl().safe_substitute(
              charge=0, n_mm_atoms=1000, read_guess='No',
              KPointsAndWeights=qmtmpl.KPointsAndWeights,
              calc_forces='Yes', addparam=''))
    qmtmpl = QMTmpl('orca')
    print(qmtmpl.gen_qmtmpl().safe_substitute(
              method='HF', basis='6-31G', calc_forces='EnGrad ',
              read_guess='NoAutoStart ',
              addparam='CHELPG ', nproc='8'))
    qmtmpl = QMTmpl('mopac')
    print(qmtmpl.gen_qmtmpl().safe_substitute(
              method='PM7', calc_forces='GRAD ',
              charge=0, addparam=' ESP', nproc='8'))