import importlib

MMCLASS = {'namd': "NAMD"}

MMMODULE = {'NAMD': ".namd"}


def choose_mmtool(mmSoftware):
    try:
        mm_class = MMCLASS[mmSoftware.lower()]
        mm_module = MMMODULE[mm_class]
    except:
        raise ValueError("Please choose 'namd' for mmSoftware.")

    mmtool = importlib.import_module(mm_module, package='qmhub.mmtools').__getattribute__(mm_class)

    return mmtool
