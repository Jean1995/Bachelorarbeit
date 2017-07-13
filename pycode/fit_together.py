# Vorheriges Ausführen von R.py für vier verschiedene Parameterkombis nötig (und mit dif_wq=1)

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


###



y_e_22, y_tau_22, y_mu_22 = np.genfromtxt('difwqges_22.txt', unpack=True)
y_e_23, y_tau_23, y_mu_23 = np.genfromtxt('difwqges_23.txt', unpack=True)
y_e_32, y_tau_32, y_mu_32 = np.genfromtxt('difwqges_32.txt', unpack=True)
y_e_33, y_tau_33, y_mu_33 = np.genfromtxt('difwqges_33.txt', unpack=True)

qq_plot_tau = np.linspace(m_tau**2, (m_b-m_d)**2 , 300)
qq_plot_e = np.linspace(m_e**2, (m_b-m_d)**2 , 300)
qq_plot_mu = np.linspace(m_mu**2, (m_b-m_d)**2 , 300)

red = 1/(10**(-15) )#* 10**9 * const.eV)


#Elektronen
plt.plot(z_from_qq(qq_plot_e) ,y_e_22*red, label=r'$N_+=1,\: N_0 = 1$.', color='y')
plt.plot(z_from_qq(qq_plot_e) ,y_e_23*red, label=r'$N_+=1,\: N_0 = 2$.', color='g')
plt.plot(z_from_qq(qq_plot_e) ,y_e_32*red, label=r'$N_+=2,\: N_0 = 1$.', color='b')
plt.plot(z_from_qq(qq_plot_e) ,y_e_33*red, label=r'$N_+=2,\: N_0 = 2$.', color='r')

plt.ylabel(r'$\frac{d \Gamma}{d q^2} \left(\overline{B} \to D e \nu_e \right) \,/\, \left( \num{e-15} \si{\giga \electronvolt\tothe{-1}} \right)$')
plt.xlabel(r'$z$')
plt.legend(loc='best', prop={'size':20})
plt.tight_layout()
plt.savefig('plot_diff_wq_ges_e.pdf') #fancy
plt.clf()

plt.plot(z_from_qq(qq_plot_e) ,y_mu_22*red, label=r'$N_+=1,\: N_0 = 1$.', color='y')
plt.plot(z_from_qq(qq_plot_e) ,y_mu_23*red, label=r'$N_+=1,\: N_0 = 2$.', color='g')
plt.plot(z_from_qq(qq_plot_e) ,y_mu_32*red, label=r'$N_+=2,\: N_0 = 1$.', color='b')
plt.plot(z_from_qq(qq_plot_e) ,y_mu_33*red, label=r'$N_+=2,\: N_0 = 2$.', color='r')
plt.ylabel(r'$\frac{d \Gamma}{d q^2} \left(\overline{B} \to D \mu \nu_{\mu} \right) \,/\, \left( \num{e-15} \si{\giga \electronvolt\tothe{-1}} \right)$')
plt.xlabel(r'$z$')
plt.legend(loc='best', prop={'size':20})
plt.tight_layout()
plt.savefig('plot_diff_wq_ges_mu.pdf') #fancy
plt.clf()

plt.plot(z_from_qq(qq_plot_e) ,y_tau_22*red, label=r'$N_+ =1, \:N_0 = 1$.', color='y')
plt.plot(z_from_qq(qq_plot_e) ,y_tau_23*red, label=r'$N_+ =1, \:N_0 = 2$.', color='g')
plt.plot(z_from_qq(qq_plot_e) ,y_tau_32*red, label=r'$N_+ =2, \:N_0 = 1$.', color='b')
plt.plot(z_from_qq(qq_plot_e) ,y_tau_33*red, label=r'$N_+ =2, \:N_0 = 2$.', color='r')
plt.ylabel(r'$\frac{\mathrm{d} \Gamma}{\mathrm{d} q^2} \left(\overline{B} \to D \tau \nu_{\tau} \right) \,/\, \left( \num{e-15} \si{\giga \electronvolt\tothe{-1}} \right)$')
plt.xlabel(r'$z$')
plt.legend(loc='best', prop={'size':20})
plt.tight_layout()
plt.savefig('plot_diff_wq_ges_tau.pdf') #fancy
plt.clf()
