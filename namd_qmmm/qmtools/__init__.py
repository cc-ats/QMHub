from .qchem import QChem
from .dftb import DFTB
from .orca import ORCA
from .mopac import MOPAC
from .psi4 import PSI4

__all__ = ['choose_qmtool']

QMTOOLS = ['QChem', 'DFTB', 'ORCA', 'MOPAC', 'PSI4']

def choose_qmtool(qmSoftware):
    for qmtool in QMTOOLS:
        qmtool = globals()[qmtool]
        if qmtool.check_software(qmSoftware):
            return qmtool
    raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac', or 'psi4' for qmSoftware.")
