import importlib

QMCLASS = {'q-chem': "QChem", 'dftb+': "DFTB", 'orca': "ORCA",
           'mopac': "MOPAC", 'psi4': "PSI4"}

QMMODULE = {'q-chem': ".qchem", 'dftb+': ".dftb", 'orca': ".orca",
            'mopac': ".mopac", 'psi4': ".psi4"}


def choose_qmtool(qmSoftware):
    try:
        qm_class = QMCLASS[qmSoftware.lower()]
        qm_module = QMMODULE[qmSoftware.lower()]
    except:
        raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac', or 'psi4' for qmSoftware.")

    qmtool = importlib.import_module(qm_module, package='namd_qmmm.qmtools').__getattribute__(qm_class)
    return qmtool
