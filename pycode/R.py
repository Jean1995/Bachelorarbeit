import fit2
from scipy.integrate import quad
import numpy as np
from numpy.linalg import inv
import scipy.constants as const
from numpy import random
from table import (
    make_SI,
    write,
    make_table,
)
from uncertainties import ufloat

from fit2 import N1, N2, m_p, m_0, m_b, m_d, m_e, m_tau, m_mu, m_bottom, m_charm, V_cb, R_exp, m_p_s, m_0_s, m_b_s, m_d_s, m_e_s, m_tau_s, m_mu_s, m_bottom_s, m_charm_s, V_cb_s, R_exp_s
from params import eta, G_f, plot_difwq, R_exp, R_exp_s

a_mc = fit2.a_mc
m_mc = fit2.m_mc

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.set_cmap('Set2')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.use_locale'] = True # kommata
plt.rcParams['text.latex.preamble'] = ['\\usepackage[locale=DE,separate-uncertainty=true,per-mode=symbol-or-fraction,]{siunitx} \\DeclareMathSymbol{,}{\mathord}{letters}{"3B}']
plt.rc('font',family='Latin Modern')

### Funktionen

def z(w):
    return (np.sqrt(w+1) - np.sqrt(2)) / (np.sqrt(w+1) + np.sqrt(2))

def qq(z):
    return m_b**2 + m_d**2 - 2*m_b*m_d * (z**2 + 6*z + 1)/(z-1)**2

def z_from_qq(qq):
    w = (m_b**2 + m_d**2 - qq)/(2*m_b*m_d)
    return z(w)

def f(z,n,m):
    return z**n/(1 - qq(z)/m**2)

z_max = z_from_qq(0)


def f_p(z, a, m_p_tmp):
    '''
        Zu beachten: Die Reihenfolge der Parameter ist (a_+0, a+1, a+2, ... a+N-1, a_00, a01, a02, ... a0N-1)
    '''
    tmp = 0
    for n in range(N1):
        tmp = tmp + a[n] * f(z, n, m_p_tmp)
    return tmp

def f_0(z, a, m_0_tmp):
    ''' '''
    tmp = 0
    for n in range(N2):
        tmp = tmp + a[n+N1] * f(z, n, m_0_tmp)
    return tmp

def dif_wq(qq, m_l, a, m_b_tmp, m_d_tmp, m_p_tmp, m_0_tmp):
    #beachte die gekürzten Faktoren: (eta^2 G_f^2 V_cb m_b)/(192 pi^3)
    r = m_d_tmp/m_b_tmp
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    c_plus = lambd/m_b_tmp**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    return np.sqrt(lambd) * (1 - m_l**2/qq)**2 * (c_plus * f_p(z_from_qq(qq), a, m_p_tmp)**2 + c_null * f_0(z_from_qq(qq), a, m_0_tmp)**2 )

def dif_wq_complete(qq, m_l, a, m_b_tmp, m_d_tmp, V_cb_tmp, m_p_tmp, m_0_tmp):
    r = m_d_tmp/m_b_tmp
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    c_plus = lambd/m_b_tmp**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    vorfaktor = eta**2 * G_f**2 * V_cb_tmp**2 * m_b_tmp * np.sqrt(lambd) * 1/(192*np.pi**3) * (1 - m_l**2 / qq)**2
    return vorfaktor * (c_plus * f_p(z_from_qq(qq), a, m_p_tmp)**2 + c_null * f_0(z_from_qq(qq), a, m_0_tmp)**2 )


###

R_values = np.array([]) # leeres array für Werte

for a, m in zip(a_mc, m_mc):
    tot_wq_e = quad(dif_wq, m[4]**2, (m[2]-m[3])**2, args=(m[4], a, m[2], m[3], m[0], m[1]))[0]
    tot_wq_tau = quad(dif_wq, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[0], m[1]))[0]
    tot_wq_mu = quad(dif_wq, m[6]**2, (m[2]-m[3])**2, args=(m[6], a, m[2], m[3], m[0], m[1]))[0]
    #print(tot_wq_e)
    R_values = np.append(R_values, 2*tot_wq_tau/(tot_wq_e + tot_wq_mu))

R_error = np.std(R_values)
R_mean = np.mean(R_values)


write('R_si' + str(N1) + str(N2) +  '.tex', make_SI(ufloat(R_mean,R_error), r'', figures=2))
print("R =",R_mean, "+-", R_error)
if N1==3 and N2==3: # ugly workaround
    write('R_' + str(N1) + str(N2) + '.tex', make_table([[ufloat(R_mean, R_error)]], [2]))
else:
    write('R_' + str(N1) + str(N2) + '.tex', make_table([[ufloat(R_mean, R_error)]], [1]))

# Abweichung R_exp zu R_mean
write('R_abweichung_' + str(N1) + str(N2) +'.tex', make_SI(abs(R_exp - R_mean)/R_exp_s, r'', figures=1))



# R-Schlange: Integriere nur von (m_b-m_d)**2 bis zu m_tau^2, um die größer werdenden Fehler des dif. WQ zu beheben
R_values_schlange = np.array([])

for a, m in zip(a_mc, m_mc):
    tot_wq_e_schlange = quad(dif_wq, m[5]**2, (m[2]-m[3])**2, args=(m[4], a, m[2], m[3], m[0], m[1]))[0]
    tot_wq_tau_schlange = quad(dif_wq, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[0], m[1]))[0]
    tot_wq_mu_schlange = quad(dif_wq, m[5]**2, (m[2]-m[3])**2, args=(m[6], a, m[2], m[3], m[0], m[1]))[0]
    #print(tot_wq_e)
    R_values_schlange = np.append(R_values_schlange, 2*tot_wq_tau_schlange/(tot_wq_e_schlange + tot_wq_mu_schlange))

R_error_schlange = np.std(R_values_schlange)
R_mean_schlange = np.mean(R_values_schlange)

print("R~", R_mean_schlange, "+-", R_error_schlange)
write('Rschlange_si' + str(N1) + str(N2) +  '.tex', make_SI(ufloat(R_mean_schlange,R_error_schlange), r'', figures=2))
write('Rschlange_' + str(N1) + str(N2) + '.tex', make_table([[ufloat(R_mean_schlange, R_error_schlange)]], [1]))

### Differentieller Wirkungsquerschnitt Elektronen / Tauonen

if plot_difwq != 0:

    qq_plot_tau = np.linspace(m_tau**2, (m_b-m_d)**2 , 300)
    qq_plot_e = np.linspace(m_e**2, (m_b-m_d)**2 , 300)
    qq_plot_mu = np.linspace(m_mu**2, (m_b-m_d)**2 , 300)
    dif_wq_val_e = np.array([])
    dif_wq_val_e_up = np.array([])
    dif_wq_val_e_down = np.array([])
    dif_wq_val_tau = np.array([])
    dif_wq_val_tau_up = np.array([])
    dif_wq_val_tau_down = np.array([])
    dif_wq_val_mu = np.array([])
    dif_wq_val_mu_up = np.array([])
    dif_wq_val_mu_down = np.array([])

    for qq_tmp in qq_plot_e:
        tmp_e = np.array([])
        for a, m in zip(a_mc, m_mc):
            tmp_e = np.append(tmp_e, dif_wq_complete(qq_tmp, m[4], a, m[2], m[3], m[9], m[0], m[1]))
        tmp_mean_e = np.mean(tmp_e)
        tmp_std_e = np.std(tmp_e)
        dif_wq_val_e = np.append(dif_wq_val_e, tmp_mean_e)
        dif_wq_val_e_up = np.append(dif_wq_val_e_up, tmp_mean_e + tmp_std_e)
        dif_wq_val_e_down = np.append(dif_wq_val_e_down, tmp_mean_e - tmp_std_e)

    for qq_tmp in qq_plot_tau:
        tmp_tau = np.array([])
        for a, m in zip(a_mc,m_mc):
            tmp_tau = np.append(tmp_tau, dif_wq_complete(qq_tmp, m[5], a, m[2], m[3], m[9], m[0], m[1]))
        tmp_mean_tau = np.mean(tmp_tau)
        tmp_std_tau = np.std(tmp_tau)
        dif_wq_val_tau = np.append(dif_wq_val_tau, tmp_mean_tau)
        dif_wq_val_tau_up = np.append(dif_wq_val_tau_up, tmp_mean_tau + tmp_std_tau)
        dif_wq_val_tau_down = np.append(dif_wq_val_tau_down, tmp_mean_tau - tmp_std_tau)

    for qq_tmp in qq_plot_mu:
        tmp_mu = np.array([])
        for a, m in zip(a_mc, m_mc):
            tmp_mu = np.append(tmp_mu, dif_wq_complete(qq_tmp, m[6], a, m[2], m[3], m[9], m[0], m[1]))
        tmp_mean_mu = np.mean(tmp_mu)
        tmp_std_mu = np.std(tmp_mu)
        dif_wq_val_mu = np.append(dif_wq_val_mu, tmp_mean_mu)
        dif_wq_val_mu_up = np.append(dif_wq_val_mu_up, tmp_mean_mu + tmp_std_mu)
        dif_wq_val_mu_down = np.append(dif_wq_val_mu_down, tmp_mean_mu - tmp_std_mu)


    red = 1/(10**(-15) )#* 10**9 * const.eV)

    plt.plot(z_from_qq(qq_plot_e) ,dif_wq_val_e*red, label=r'Dif. Zerfallsbreite, $l = e$.', color='b')
    plt.fill_between(z_from_qq(qq_plot_e), dif_wq_val_e_up*red,  dif_wq_val_e_down*red, interpolate=True, alpha=0.3, color='b',linewidth=0.0)

    plt.plot(z_from_qq(qq_plot_tau) ,dif_wq_val_tau*red, label=r'Dif. Zerfallsbreite, $l = \tau$.', color='r')
    plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_val_tau_up*red,  dif_wq_val_tau_down*red, interpolate=True, alpha=0.3, color='r',linewidth=0.0)

    plt.plot(z_from_qq(qq_plot_mu) ,dif_wq_val_mu*red, label=r'Dif. Zerfallsbreite, $l = \mu$.', color='g')
    plt.fill_between(z_from_qq(qq_plot_mu), dif_wq_val_mu_up*red,  dif_wq_val_mu_down*red, interpolate=True, alpha=0.3, color='g',linewidth=0.0)

    plt.ylabel(r'$\frac{d \Gamma}{d q^2} \left(B \to D l \nu_l \right) \,/\, \left( \num{e-15} \si{\giga \electronvolt\tothe{-1}} \right)$')
    plt.xlabel(r'$z$')
    plt.legend(loc='best', prop={'size':20})
    plt.tight_layout()
    plt.savefig('plot_diff_wq' + str(N1) + str(N2) + '.pdf') #fancy
    plt.clf()

    np.savetxt('difwqges_'+str(N1)+str(N2)+'.txt', np.column_stack([dif_wq_val_e, dif_wq_val_tau, dif_wq_val_mu]))
