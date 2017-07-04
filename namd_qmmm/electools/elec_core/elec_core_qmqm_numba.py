import math
import numba as nb


@nb.guvectorize([(nb.float64[:], nb.float64[:])],
                '()->()', nopython=True, target='cpu', cache=True)
def get_coulomb_esp(dij, coulomb_esp):
    if dij[0] != 0.0:
        coulomb_esp[0] = 1.0 / dij[0]
    else:
        coulomb_esp[0] = 0.0


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],
                '(m),()->(m)', nopython=True, target='cpu', cache=True)
def get_coulomb_efield(rij, dij, coulomb_efield):
    if dij[0] != 0.0:
        coulomb_efield[0] = rij[0] / dij[0]**3
        coulomb_efield[1] = rij[1] / dij[0]**3
        coulomb_efield[2] = rij[2] / dij[0]**3
    else:
        coulomb_efield[0] = 0.0
        coulomb_efield[1] = 0.0
        coulomb_efield[2] = 0.0


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:])],
                '(m),(n,m),()->()', nopython=True, target='parallel', cache=True)
def get_ewald_real_esp(rij, n, alpha, ewald_real_esp):
    ewald_real_esp[0] = 0.0

    for i in range(len(n)):
        d = math.sqrt((rij[0] + n[i, 0])**2 + (rij[1] + n[i, 1])**2 + (rij[2] + n[i, 2])**2)
        if d != 0:
            ewald_real_esp[0] += math.erfc(alpha[0] * d) / d


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:])],
                '(m),(n,m),()->(m)', nopython=True, target='parallel', cache=True)
def get_ewald_real_efield(rij, n, alpha, ewald_real_efield):
    ewald_real_efield[0] = 0.0
    ewald_real_efield[1] = 0.0
    ewald_real_efield[2] = 0.0

    for i in range(len(n)):
        r_0 = rij[0] + n[i, 0]
        r_1 = rij[1] + n[i, 1]
        r_2 = rij[2] + n[i, 2]
        d = math.sqrt(r_0**2 + r_1**2 + r_2**2)
        if d != 0:
            prod = (math.erfc(alpha[0] * d) / d**3
                    + 2 * alpha[0] * math.exp(-1 * alpha[0]**2 * d**2) / math.sqrt(math.pi) / d**2)
            ewald_real_efield[0] += prod * r_0
            ewald_real_efield[1] += prod * r_1
            ewald_real_efield[2] += prod * r_2


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:])],
                '(m),(n,m),(n)->()', nopython=True, target='parallel', cache=True)
def get_ewald_recip_esp(rij, k, fac, ewald_recip_esp):
    ewald_recip_esp[0] = 0.0

    for n in range(len(k)):
        kr = k[n, 0] * rij[0] + k[n, 1] * rij[1] + k[n, 2] * rij[2]
        ewald_recip_esp[0] += fac[n] * math.cos(kr)


@nb.guvectorize([(nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:])],
                '(m),(n,m),(n)->(m)', nopython=True, target='parallel', cache=True)
def get_ewald_recip_efield(rij, k, fac, ewald_recip_efield):
    ewald_recip_efield[0] = 0.0
    ewald_recip_efield[1] = 0.0
    ewald_recip_efield[2] = 0.0

    for n in range(len(k)):
        kr = k[n, 0] * rij[0] + k[n, 1] * rij[1] + k[n, 2] * rij[2]
        prod = fac[n] * math.sin(kr)
        ewald_recip_efield[0] += prod * k[n, 0]
        ewald_recip_efield[1] += prod * k[n, 1]
        ewald_recip_efield[2] += prod * k[n, 2]


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:])],
                '(),()->()', nopython=True, target='cpu', cache=True)
def get_ewald_self_esp(dij, alpha, ewald_self_esp):
    if dij[0] == 0.0:
        ewald_self_esp[0] = 2 * alpha[0] / math.sqrt(math.pi)
    else:
        ewald_self_esp[0] = 0.0
