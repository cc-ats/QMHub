import sys
import numpy as np

from . import mmtools
from . import qmtools


class QMMM(object):
    def __init__(self, fin, qmSoftware, qmCharge, qmMult,
                 elecMode, qmElecEmbed=True,
                 qmRefChrgs=None, qmSwitchingType=None,
                 qmCutoff=None, qmSwdist=None, 
                 postProc=False, qmReadGuess=False):
        """
        Creat a QMMM object.
        """
        self.qmSoftware = qmSoftware
        self.qmCharge = qmCharge
        self.qmMult = qmMult
        self.elecMode = elecMode
        self.qmElecEmbed = qmElecEmbed
        self.qmRefChrgs = qmRefChrgs
        self.qmCutoff = qmCutoff
        self.qmSwdist = qmSwdist
        self.postProc = postProc
        self.qmReadGuess = qmReadGuess

        # Initialize the system
        self.system = mmtools.NAMD(fin)

        # Set the refernce charges for QM atoms
        if qmRefChrgs is not None:
            self.qmRefChrgs = qmRefChrgs
        else:
            self.qmRefChrgs = self.system.qm_charge0

        # Prepare for the electrostatic model
        if self.elecMode.lower() in {'truncation', 'mmewald'}:
            self.qmPBC = False
            self.system.qm_charge_me = self.qmRefChrgs
        elif self.elecMode.lower() == 'qmewald':
            self.qmPBC = True
            self.system.qm_charge_me = np.zeros(self.system.n_qm_atoms)
        else:
            raise ValueError("Only 'truncation', 'mmewald', and 'qmewald' are supported at the moment.")

        # Prepare for QM with PBC
        if self.qmPBC:
            if self.system.n_atoms != self.system.n_real_qm_atoms + self.system.n_real_mm_atoms:
                raise ValueError("Unit cell is not complete.")

            self.split_mm_atoms(qmCutoff=self.qmCutoff)

        # Set switching function for external point charges
        if qmSwitchingType is not None:
            self.qmSwitchingType = qmSwitchingType
        else:
            self.qmSwitchingType = 'shift'

        if self.elecMode.lower() == 'mmewald':
            if not self.qmElecEmbed:
                self.qmSwitchingType = None
        elif self.elecMode.lower() == 'qmewald':
            self.qmSwitchingType = None

        # Scale the external point charges
        if self.qmSwitchingType is not None:
            self.system.get_min_distances()
            self.system.scale_charges(self.qmSwitchingType, self.qmCutoff, self.qmSwdist)

        if self.elecMode.lower() == 'qmewald':
            if not self.qmElecEmbed:
                raise ValueError("Can not use elecMode='qmewald' with qmElecEmbed=False.")
            self.system.mm_charge_qm = self.system.mm_charge
        elif self.elecMode.lower() == 'mmewald':
            self.system.mm_charge_mm = self.system.mm_charge
            if self.qmElecEmbed:
                self.system.mm_charge_qm = self.system.mm_charge_scaled
            else:
                self.system.mm_charge_qm = np.zeros(self.system.n_mm_atoms)
        elif self.elecMode.lower() == 'truncation':
            if self.qmElecEmbed:
                self.system.mm_charge_qm = self.system.mm_charge_scaled
            else:
                self.system.mm_charge_mm = self.system.mm_charge_scaled
                self.system.mm_charge_qm = np.zeros(self.system.n_mm_atoms)

        # Initialize the QM system
        self.qm = qmtools.choose_qmtool(self.qmSoftware)(self.system, self.qmCharge, self.qmMult, self.qmPBC)

        if self.postProc:
            self.qm.calc_forces = False
        else:
            self.qm.calc_forces = True

        if self.qmReadGuess and not self.system.step == 0:
            self.qm.read_guess = True
        else:
            self.qm.read_guess = False

    def run_qm(self, **kwargs):
        """Run QM calculation."""
        self.qm.get_qm_params(**kwargs)
        self.qm.gen_input()
        self.qm.run()
        if self.qm.exitcode != 0:
            sys.exit(self.qm.exitcode)

    def dry_run_qm(self, **kwargs):
        """Generate input file without running QM calculation."""
        if self.postProc:
            self.qm.get_qm_params(**kwargs)
            self.qm.gen_input()
        else:
            raise ValueError("dryrun_qm() can only be used with postProc=True.""")

    def parse_output(self):
        """Parse the output of QM calculation."""
        if self.postProc:
            self.system.parse_output(self.qm)
        elif hasattr(self.qm, 'exitcode'):
            if self.qm.exitcode == 0:
                self.system.parse_output(self.qm)

                if self.qmElecEmbed and not self.qmPBC:
                    self.system.corr_elecembed()

                if not self.qmElecEmbed or self.elecMode.lower() == 'mmewald':
                    self.system.corr_mechembed()

                self.system.corr_virt_mm_atoms()
            else:
                raise ValueError("QM calculation did not finish normally.")
        else:
            raise ValueError("Need to run_qm() first.")

    def save_results(self):
        """Save the results of QM calculation to file."""
        if hasattr(self.system, 'qm_energy'):
            self.system.save_results()
        else:
            raise ValueError("Need to parse_output() first.")

    def save_charges(self):
        """Save the QM and MM charges to file (for debugging only)."""
        system_scale = np.zeros(self.system.n_atoms)
        system_dij_min = np.zeros(self.system.n_atoms)
        system_charge = np.zeros(self.system.n_atoms)

        if hasattr(self.system, 'charge_scale'):
            system_scale[self.system.mm_index[0:self.system.n_real_mm_atoms]] = self.system.charge_scale
            system_dij_min[self.system.mm_index[0:self.system.n_real_mm_atoms]] = self.system.dij_min
        else:
            system_scale[self.system.mm_index[0:self.system.n_real_mm_atoms]] += 1
        system_charge[self.system.mm_index[0:self.system.n_real_mm_atoms]] = self.system.mm_charge_qm[0:self.system.n_real_mm_atoms]

        system_scale[self.system.qm_index[0:self.system.n_real_qm_atoms]] = np.ones(self.system.n_real_qm_atoms)
        system_charge[self.system.qm_index[0:self.system.n_real_qm_atoms]] = self.system.qm_charge_me[0:self.system.n_real_qm_atoms]

        np.save(self.qm.basedir + "system_scale", system_scale)
        np.save(self.qm.basedir + "system_dij_min", system_dij_min)
        np.save(self.qm.basedir + "system_charge", system_charge)
