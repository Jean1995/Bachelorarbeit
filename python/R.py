import fit2
from scipy.integrate import quad
import numpy as np
from numpy.linalg import inv
import scipy.constants as const
from numpy import random

from params import m_tau, N1, N2, eta, G_f, V_cb, m_e, m_mu, m_p, m_0, m_b, m_d, plot_difwq
import matplotlib.pyplot as plt



a_mc = fit2.a_mc

m_p = fit2.m_p
m_0 = fit2.m_0
m_b = fit2.m_b
m_d = fit2.m_d

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


def f_p(z, a):
    '''
        Zu beachten: Die Reihenfolge der Parameter ist (a_+0, a+1, a+2, ... a+N-1, a_00, a01, a02, ... a0N-1)
    '''
    tmp = 0
    for n in range(N1):
        tmp = tmp + a[n] * f(z, n, m_p)
    return tmp

def f_0(z, a):
    ''' '''
    tmp = 0
    for n in range(N2):
        tmp = tmp + a[n+N1] * f(z, n, m_0)
    return tmp

def dif_wq(qq, m_l, a):
    #beachte die gekürzten Faktoren: (eta^2 G_f^2 V_cb m_b)/(192 pi^3)
    r = m_d/m_b
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    c_plus = lambd/m_b**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    return np.sqrt(lambd) * (1 - m_l**2/qq)**2 * (c_plus * f_p(z_from_qq(qq), a)**2 + c_null * f_0(z_from_qq(qq), a)**2 )

def dif_wq_complete(qq, m_l, a):
    r = m_d/m_b
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    c_plus = lambd/m_b**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    vorfaktor = eta**2 * G_f**2 * V_cb**2 * m_b * np.sqrt(lambd) * 1/(192*np.pi**3) * (1 - m_l**2 / qq)**2
    return vorfaktor * (c_plus * f_p(z_from_qq(qq), a)**2 + c_null * f_0(z_from_qq(qq), a)**2 )


###

R_values = np.array([]) # leeres array für Werte

for a in a_mc:
    tot_wq_e = quad(dif_wq, m_e**2, (m_b-m_d)**2, args=(m_e, a))[0]
    tot_wq_tau = quad(dif_wq, m_tau**2, (m_b-m_d)**2, args=(m_tau, a))[0]
    tot_wq_mu = quad(dif_wq, m_mu**2, (m_b-m_d)**2, args=(m_mu, a))[0]
    #print(tot_wq_e)
    R_values = np.append(R_values, 2*tot_wq_tau/(tot_wq_e + tot_wq_mu))

R_error = np.std(R_values)
R_mean = np.mean(R_values)

print(R_mean, "+-", R_error)

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
        for a in a_mc:
            tmp_e = np.append(tmp_e, dif_wq_complete(qq_tmp, m_e, a))
        tmp_mean_e = np.mean(tmp_e)
        tmp_std_e = np.std(tmp_e)
        dif_wq_val_e = np.append(dif_wq_val_e, tmp_mean_e)
        dif_wq_val_e_up = np.append(dif_wq_val_e_up, tmp_mean_e + tmp_std_e)
        dif_wq_val_e_down = np.append(dif_wq_val_e_down, tmp_mean_e - tmp_std_e)

    for qq_tmp in qq_plot_tau:
        tmp_tau = np.array([])
        for a in a_mc:
            tmp_tau = np.append(tmp_tau, dif_wq_complete(qq_tmp, m_tau, a))
        tmp_mean_tau = np.mean(tmp_tau)
        tmp_std_tau = np.std(tmp_tau)
        dif_wq_val_tau = np.append(dif_wq_val_tau, tmp_mean_tau)
        dif_wq_val_tau_up = np.append(dif_wq_val_tau_up, tmp_mean_tau + tmp_std_tau)
        dif_wq_val_tau_down = np.append(dif_wq_val_tau_down, tmp_mean_tau - tmp_std_tau)

    for qq_tmp in qq_plot_mu:
        tmp_mu = np.array([])
        for a in a_mc:
            tmp_mu = np.append(tmp_mu, dif_wq_complete(qq_tmp, m_mu, a))
        tmp_mean_mu = np.mean(tmp_mu)
        tmp_std_mu = np.std(tmp_mu)
        dif_wq_val_mu = np.append(dif_wq_val_mu, tmp_mean_mu)
        dif_wq_val_mu_up = np.append(dif_wq_val_mu_up, tmp_mean_mu + tmp_std_mu)
        dif_wq_val_mu_down = np.append(dif_wq_val_mu_down, tmp_mean_mu - tmp_std_mu)


        red = 1/(10**(-15) )#* 10**9 * const.eV)

        plt.plot(z_from_qq(qq_plot_e) ,dif_wq_val_e*red, label=r'Vorhersage dif. WQ. $l = e$. Paramterzahl N1 = ' + str(N1) + ' N2 = ' + str(N2))
        plt.etween(z_from_qq(qq_plot_e), dif_wq_val_e_up*red,  dif_wq_val_e_down*red, interpolate=True, alpha=0.5)

        plt.plot(z_from_qq(qq_plot_tau) ,dif_wq_val_tau*red, label=r'Vorhersage dif. WQ. $l = \tau$. Paramterzahl N1 = ' +  str(N1)+ ' N2 = ' + str(N2))
        plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_val_tau_up*red,  dif_wq_val_tau_down*red, interpolate=True, alpha=0.5)

        plt.plot(z_from_qq(qq_plot_mu) ,dif_wq_val_mu*red, label=r'Vorhersage dif. WQ. $l = \mu$. Paramterzahl N1 = ' + str(N1) +   ' N2 = ' + str(N2))
        plt.fill_between(z_from_qq(qq_plot_mu), dif_wq_val_mu_up*red,  dif_wq_val_mu_down*red, interpolate=True, alpha=0.5)

        plt.ylabel(r'$\frac{d \Gamma}{d q^2} \left(B \to D l \nu_l \right) \,/\, \num{e-15} \si{\giga \electronvolt} $')
        plt.xlabel(r'$z$')
        plt.legend(loc='best')
        plt.savefig('plot_diff_wq' + str(N1) + str(N2) + '.pdf') #fancy
        plt.clf()
