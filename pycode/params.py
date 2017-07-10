import numpy as np
import scipy.constants as const
from table import (
    make_SI,
    write,
)
from uncertainties import ufloat

#
N1 = 3 #Parameter f+
N2 = 3 #Parameter f0

plot_difwq = 1 # entscheide, ob der differentielle WQ geplottet werden soll (zeitlicher Aufwand von ca. einer Minute)
### Konstanten

m_b = 5279.26 *10**(-3) #* 10**6 * const.electron_volt#/const.c**2
m_b_s = 0.17 * 10**(-3)
#^ Quelle: PDG '14
m_d = 1869.61*10**(-3) #* 10**6 * const.electron_volt#/const.c**2
m_d_s = 0.10 * 10**(-3)
#^ Quelle: PDG '14

m_p = 6329*10**(-3) #* 10**6 * const.electron_volt#/const.c**2 # Richtige Resonanzmasse hier einfügen. Ist bisher nur eine zufällige aus 1606.08030, S. 5
m_p_s = 3 * 10**(-3)
m_0 = 6716*10**(-3) #* 10**6 * const.electron_volt#/const.c**2
#m_0 = 6329*10**(-3) #* 10**6 * const.electron_volt#/const.c**2
m_0_s = 0
#^ Quelle 1606.08030

write('mp.tex', make_SI(ufloat(m_p,m_p_s), r'\giga\electronvolt', figures=1))
write('m0.tex', make_SI(m_0, r'\giga\electronvolt', figures=3))


m_e = 0.510998928 * 10 **(-3)
m_e_s = 0.000000011 * 10**(-3)
m_tau = 1776.82 * 10**(-3)
m_tau_s = 0.16 * 10**(-3)
m_mu = 105.6583715 * 10**(-3)
m_mu_s = 0.0000035 * 10**(-3)

m_bottom = 4180 * 10**(-3)
m_bottom_s = 10 * 10**(-3)
m_charm = 1275 * 10**(-3)
m_charm_s = 25 * 10**(-3)
#^ Quelle PDG (alles obrigen leptonen/quarks)

write('m_bottom.tex', make_SI(ufloat(m_bottom,m_bottom_s), r'\mega\electronvolt', figures=2))
write('m_charm.tex', make_SI(ufloat(m_charm,m_charm_s), r'\mega\electronvolt', figures=2))

eta = 1.0066 #src https://arxiv.org/pdf/1606.08030.pdf
G_f = 1.1663787*10**(-5) #* 1/(10**9 * const.electron_volt)**2  #* (const.hbar * const.c)**3
V_cb = 40.49*10**(-3)
V_cb_s = 0.97*10**(-3)
#^ Quelle: 1703.06124
write('V_cb.tex', make_SI(ufloat(V_cb,V_cb_s)*1000, r'','e-3', figures=2))


### Gesamtdaten

w_roh = np.array([1, 1.08, 1.16]) # Werte für w
lattice_roh = np.array([1.1994, 1.0941, 1.0047, 0.9026, 0.8609, 0.8254]) # Latticewerte
s_l = np.array([0.0095, 0.0104, 0.0123, 0.0072, 0.0077, 0.0094]) # Abweichungen Lattice

#corr_mat = np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
corr_mat = np.array([[1, 0.9674, 0.8812, 0.8290, 0.8533, 0.8032],[0.9674, 1, 0.9523, 0.8241, 0.8992, 0.8856],[0.8812, 0.9523, 1, 0.7892, 0.8900, 0.9530],[0.8290, 0.8241, 0.7892, 1, 0.9650, 0.8682],[0.8533, 0.8992, 0.8900, 0.9650, 1, 0.9519], [0.8032, 0.8856, 0.9530, 0.8682, 0.9519, 1]]) #Korellationsmatrix
V = np.zeros((len(lattice_roh), len(lattice_roh))) # Kovarianzmatrix
for i in range(len(lattice_roh)):
    for j in range(len(lattice_roh)):
        V[i,j] = corr_mat[i,j] * s_l[i] * s_l[j]



### Weitere Daten

R_babar_m = 0.440
R_babar_stat = 0.058
R_babar_syst = 0.042
R_belle_m = 0.375
R_belle_stat = 0.064
R_belle_syst = 0.026

R_babar = ufloat(R_babar_m, np.sqrt(R_babar_stat**2 + R_babar_syst**2))
R_belle = ufloat(R_belle_m, np.sqrt(R_belle_stat**2 + R_belle_syst**2))
#systematischer Fehler und statistischer Fehler quadratisch addiert

R_exp_mean = (R_babar/R_babar.s**2 + R_belle/R_belle.s**2)/(1/R_babar.s**2 + 1/R_belle.s**2)
#gewichteter Mittelwert

R_exp = R_exp_mean.n
R_exp_s = R_exp_mean.s

write('R_exp_mean.tex', make_SI(R_exp_mean, r'', figures=2))
np.savetxt('R_babar_0.tex', ['\\num{' + str(R_babar_m) + '}'], fmt='%s')
np.savetxt('R_babar_1.tex', ['\\num{' + str(R_babar_stat) + '}'], fmt='%s')
np.savetxt('R_babar_2.tex', ['\\num{' + str(R_babar_syst) + '}'], fmt='%s')
np.savetxt('R_belle_0.tex', ['\\num{' + str(R_belle_m) + '}'], fmt='%s')
np.savetxt('R_belle_1.tex', ['\\num{' + str(R_belle_stat) + '}'], fmt='%s')
np.savetxt('R_belle_2.tex', ['\\num{' + str(R_belle_syst) + '}'], fmt='%s')

R_quelle = ufloat(0.299,0.011)
write('R_quelle.tex', make_SI(R_quelle, r'', figures=2))
