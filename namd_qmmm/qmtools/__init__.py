import importlib

QMCLASS = {'q-chem': "QChem", 'dftb+': "DFTB", 'orca': "ORCA",
           'mopac': "MOPAC", 'psi4': "PSI4"}

QMMODULE = {'QChem': ".qchem", 'DFTB': ".dftb", 'ORCA': ".orca",
            'MOPAC': ".mopac", 'PSI4': ".psi4"}


def choose_qmtool(qmSoftware):
    try:
        qm_class = QMCLASS[qmSoftware.lower()]
        qm_module = QMMODULE[qm_class]
    except:
        raise ValueError("Please choose 'q-chem', 'dftb+', 'orca', 'mopac', or 'psi4' for qmSoftware.")

    qmtool = importlib.import_module(qm_module, package='namd_qmmm.qmtools').__getattribute__(qm_class)

    return qmtool
