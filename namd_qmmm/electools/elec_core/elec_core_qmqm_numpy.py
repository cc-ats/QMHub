import math
import numpy as np
from scipy import special


def _get_coulomb_esp(dij):
    if dij != 0.0:
        return 1.0 / dij
    else:
        return 0.0


get_coulomb_esp = np.vectorize(_get_coulomb_esp, signature='()->()')


def _get_coulomb_efield(rij, dij):
    if dij != 0.0:
        return rij / dij**3
    else:
        return np.zeros(3, dtype=float)


get_coulomb_efield = np.vectorize(_get_coulomb_efield, signature='(m),()->(m)')


def _get_ewald_real_esp(rij, n, alpha):
    d = np.sqrt(np.sum((rij + n)**2, axis=1))
    return (special.erfc(alpha * d) / d).sum()


get_ewald_real_esp = np.vectorize(_get_ewald_real_esp, signature='(m),(n,m),()->()')


def _get_ewald_real_efield(rij, n, alpha):
    r = rij + n
    d = np.sqrt(np.sum(r**2, axis=1))
    prod = (special.erfc(alpha * d) / d**3
            + 2 * alpha * np.exp(-1 * alpha**2 * d**2) / math.sqrt(math.pi) / d**2)
    return (prod[:, np.newaxis] * r).sum(axis=0)


get_ewald_real_efield = np.vectorize(_get_ewald_real_efield, signature='(m),(n,m),()->(m)')


def _get_ewald_recip_esp(rij, k, fac):
    kr = np.dot(k, rij)
    return np.sum(fac * np.cos(kr))


get_ewald_recip_esp = np.vectorize(_get_ewald_recip_esp, signature='(m),(n,m),(n)->()')


def _get_ewald_recip_efield(rij, k, fac):

    kr = np.dot(k, rij)
    prod = fac * np.sin(kr)
    return (prod[:, np.newaxis] * k).sum(axis=0)


get_ewald_recip_efield = np.vectorize(_get_ewald_recip_efield, signature='(m),(n,m),(n)->(m)')


def _get_ewald_self_esp(dij, alpha):
    if dij == 0.0:
        return 2 * alpha / math.sqrt(math.pi)
    else:
        return 0.0


get_ewald_self_esp = np.vectorize(_get_ewald_self_esp, signature='(),()->()')
