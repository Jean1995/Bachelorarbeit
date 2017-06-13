import numpy as np
from params import m_tau, N1, N2, eta, G_f, V_cb, m_e, m_mu, m_p, m_0, m_b, m_d, plot_difwq, R_exp, R_exp_s, m_bottom, m_charm
from scipy.integrate import quad
from numpy import random

from fit2 import a_mc, m_mc # Importiere die Monte Carlo Parameter

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib.pyplot as plt

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

def Hss(qq, a, m_b_tmp, m_d_tmp, m_bottom_tmp, m_charm_tmp, m_0_tmp):
    return (m_b_tmp**2 - m_d_tmp**2)/(m_bottom_tmp - m_charm_tmp) * f_0(z_from_qq(qq), a, m_0_tmp)

def Hvts(qq, a, m_b_tmp, m_d_tmp, m_0_tmp):
    return (m_b_tmp**2 - m_d_tmp**2)/np.sqrt(qq) * f_0(z_from_qq(qq), a, m_0_tmp)

def Hv0s(qq, a, m_b_tmp, m_d_tmp, m_p_tmp):
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    return np.sqrt(lambd/qq) * f_p(z_from_qq(qq), a, m_p_tmp)


### BSM Funktionen aus Paper https://arxiv.org/pdf/1309.0301.pdf (8)

def A_s(qq, m_l, a, m_b_tmp, m_d_tmp, V_cb_tmp, m_bottom_tmp, m_charm_tmp, m_0_tmp):
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    return (eta * G_f**2 * V_cb_tmp**2)/(192 * np.pi**3 * m_b_tmp**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3/2 * Hss(qq, a, m_b_tmp, m_d_tmp, m_bottom_tmp, m_charm_tmp, m_0_tmp)**2

def A_vs(qq, m_l, a, m_b_tmp, m_d_tmp, V_cb_tmp, m_bottom_tmp, m_charm_tmp, m_0_tmp):
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    return (eta * G_f**2 * V_cb_tmp**2)/(192 * np.pi**3 * m_b_tmp**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3 * m_l/np.sqrt(qq) * Hss(qq, a, m_b_tmp, m_d_tmp, m_bottom_tmp, m_charm_tmp, m_0_tmp) * Hvts(qq, a, m_b_tmp, m_d_tmp, m_0_tmp)

def wq(qq, m_l, a, m_b_tmp, m_d_tmp, V_cb_tmp, m_0_tmp, m_p_tmp):
    lambd = (qq - m_b_tmp**2 - m_d_tmp**2)**2 - 4*m_b_tmp**2*m_d_tmp**2
    return (eta * G_f**2 * V_cb_tmp**2)/(192 * np.pi**3 * m_b_tmp**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * ( (1 + m_l**2 /(2*qq))* Hv0s(qq, a, m_b_tmp, m_d_tmp, m_p_tmp)**2 + 3/2 * m_l**2/qq * Hvts(qq, a, m_b_tmp, m_d_tmp, m_0_tmp)**2 )


def I_fc(R, As, Avs, Br_mu, Br_e, Br_tau, Rexp):
    return np.sqrt( 1/As * ( Rexp/2 * ( Br_mu + Br_e ) - Br_tau - As*R**2 - Avs * R ) )

def R_fc_up(I, As, Avs, Br_mu, Br_e, Br_tau, Rexp):
    return - Avs/(2*As) + np.sqrt( Avs**2/(4*As**2) + Rexp/(2*As) * (Br_mu + Br_e) - Br_tau/As - I**2 )

def R_fc_down(I, As, Avs, Br_mu, Br_e, Br_tau, Rexp):
    return - Avs/(2*As) - np.sqrt( Avs**2/(4*As**2) + Rexp/(2*As) * (Br_mu + Br_e) - Br_tau/As - I**2 )

As_values = []
Avs_values = []
wq_tau_values = []
wq_e_values = []
wq_mu_values = []
R_exp_values = random.normal(R_exp, R_exp_s, np.shape(a_mc)[0] ) # die erste Komponente von a_mc sagt, wie wieve Samples es gibt


for a, m in zip(a_mc, m_mc):
    tmp_as = quad(A_s, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[7], m[8], m[1]))[0]
    tmp_avs = quad(A_vs, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[7], m[8], m[1]))[0]
    tmp_wq_tau = quad(wq, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[1], m[0]))[0]
    tmp_wq_e = quad(wq, m[4]**2, (m[2]-m[3])**2, args=(m[4], a, m[2], m[3], m[9], m[1], m[0]))[0]
    tmp_wq_mu = quad(wq, m[6]**2, (m[2]-m[3])**2, args=(m[6], a, m[2], m[3], m[9], m[1], m[0]))[0]
    As_values.append(tmp_as)
    Avs_values.append(tmp_avs)
    wq_tau_values.append(tmp_wq_tau)
    wq_e_values.append(tmp_wq_e)
    wq_mu_values.append(tmp_wq_mu)

As_values = np.array(As_values)
Avs_values = np.array(Avs_values)
wq_tau_values = np.array(wq_tau_values)
wq_e_values = np.array(wq_e_values)
wq_mu_values = np.array(wq_mu_values)
R_exp_values = np.array(R_exp_values)

As_mean = np.mean(As_values)
As_std = np.std(As_values)
Avs_mean = np.mean(Avs_values)
Avs_std = np.std(Avs_values)
wq_tau_mean = np.mean(wq_tau_values)
wq_e_mean = np.mean(wq_e_values)
wq_mu_mean = np.mean(wq_mu_values)

r_sort = []

for As, Avs, wq_mu, wq_e, wq_tau, R_exp in zip(As_values, Avs_values, wq_mu_values, wq_e_values, wq_tau_values, R_exp_values):
    sort = - Avs_mean / (2*As_mean) + np.sqrt( Avs_mean**2/(4*As_mean**2) + R_exp/(2*As_mean) * ( wq_e_mean + wq_mu_mean ) -    wq_tau_mean/As_mean )
    r_sort.append(sort)


index_start = int((len(r_sort)-len(r_sort)*0.68)/2)
index_length = int(len(r_sort) * 0.68)
index_right = np.argsort(r_sort)[index_start : index_start+index_length]


#R_min_1 = - Avs_values[index_right[0]] / (2*As_values[index_right[0]]) - np.sqrt( #Avs_values[index_right[0]]**2/(4*As_values[index_right[0]]**2) + R_exp/(2*As_values[index_right[0]]) * ( wq_e_values[index_right[0]] + #wq_mu_values[index_right[0]] ) - wq_tau_values[index_right[0]]/As_values[index_right[0]] )
#R_min_2 = - Avs_values[index_right[-1]] / (2*As_values[index_right[-1]]) - np.sqrt( #Avs_values[index_right[-1]]**2/(4*As_values[index_right[-1]]**2) + R_exp/(2*As_values[index_right[-1]]) * ( wq_e_values[index_right[-1]] + #wq_mu_values[index_right[-1]] ) - wq_tau_values[index_right[-1]]/As_values[index_right[-1]] )
#R_max_1 = - Avs_values[index_right[0]] / (2*As_values[index_right[0]]) + np.sqrt( #Avs_values[index_right[0]]**2/(4*As_values[index_right[0]]**2) + R_exp/(2*As_values[index_right[0]]) * ( wq_e_values[index_right[0]] + #wq_mu_values[index_right[0]] ) - wq_tau_values[index_right[0]]/As_values[index_right[0]] )
#R_max_2 = - Avs_values[index_right[-1]] / (2*As_values[index_right[-1]]) + np.sqrt( #Avs_values[index_right[-1]]**2/(4*As_values[index_right[-1]]**2) + R_exp/(2*As_values[index_right[-1]]) * ( wq_e_values[index_right[-1]] + #wq_mu_values[index_right[-1]] ) - wq_tau_values[index_right[-1]]/As_values[index_right[-1]] )

R_plot_min = np.linspace(-2, 1, 10000)
R_plot_max = np.linspace(-2, 1, 10000)
#R_plot_min = np.linspace(R_min_1, R_max_1, 10000)
#R_plot_max = np.linspace(R_min_2, R_max_2, 10000)


fig, ax = plt.subplots()


ax.fill_between(R_plot_max, -I_fc(R_plot_max, As_values[index_right[-1]], Avs_values[index_right[-1]], wq_mu_values[index_right[-1]], wq_e_values[index_right[-1]], wq_tau_values[index_right[-1]], R_exp_values[index_right[-1]]), I_fc(R_plot_max, As_values[index_right[-1]], Avs_values[index_right[-1]], wq_mu_values[index_right[-1]], wq_e_values[index_right[-1]], wq_tau_values[index_right[-1]], R_exp_values[index_right[-1]]))

ax.fill_between(R_plot_min, -I_fc(R_plot_min, As_values[index_right[0]], Avs_values[index_right[0]], wq_mu_values[index_right[0]], wq_e_values[index_right[0]], wq_tau_values[index_right[0]], R_exp_values[index_right[0]]), I_fc(R_plot_min, As_values[index_right[0]], Avs_values[index_right[0]], wq_mu_values[index_right[0]], wq_e_values[index_right[0]], wq_tau_values[index_right[0]], R_exp_values[index_right[0]]), color="white")


#plt.plot(R_plot_min, I_fc(R_plot_min, As_values[index_right[0]], Avs_values[index_right[0]], wq_mu_values[index_right[0]], #wq_e_values[index_right[0]], wq_tau_values[index_right[0]], R_exp_values[index_right[0]]), 'r' )
#plt.plot(R_plot_max, I_fc(R_plot_max, As_values[index_right[-1]], Avs_values[index_right[-1]], wq_mu_values[index_right[-1]], #wq_e_values[index_right[-1]], wq_tau_values[index_right[-1]], R_exp_values[index_right[-1]]), 'r' )
#plt.plot(R_plot_min, -I_fc(R_plot_min, As_values[index_right[0]], Avs_values[index_right[0]], wq_mu_values[index_right[0]], #wq_e_values[index_right[0]], wq_tau_values[index_right[0]], R_exp_values[index_right[0]]), 'r' )
#plt.plot(R_plot_max, -I_fc(R_plot_max, As_values[index_right[-1]], Avs_values[index_right[-1]], wq_mu_values[index_right[-1]], #wq_e_values[index_right[-1]], wq_tau_values[index_right[-1]], R_exp_values[index_right[-1]]), 'r' )



plt.show()
### Plotte den coolen Kreis über R

#R_min = - Avs_mean / (2*As_mean) - np.sqrt( Avs_mean**2/(4*As_mean**2) + R_exp/(2*As_mean) * ( wq_e_mean + wq_mu_mean ) - #wq_tau_mean/As_mean )
#R_max = - Avs_mean / (2*As_mean) + np.sqrt( Avs_mean**2/(4*As_mean**2) + R_exp/(2*As_mean) * ( wq_e_mean + wq_mu_mean ) - #wq_tau_mean/As_mean )
#I_min = - np.sqrt( Avs_mean**2/(4*As_mean**2) + R_exp/(2*As_mean) * (wq_e_mean + wq_mu_mean) - wq_tau_mean/As_mean )
#I_max = + np.sqrt( Avs_mean**2/(4*As_mean**2) + R_exp/(2*As_mean) * (wq_e_mean + wq_mu_mean) - wq_tau_mean/As_mean )
#
#R_plot = np.linspace(R_min, R_max, 10000)
#
#plt.plot(R_plot, I_fc(R_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp), 'r' )
#plt.plot(R_plot, -I_fc(R_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp), 'r' )
#
## fehlerdinger
#
#plt.fill_between(R_plot, I_fc(R_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp+R_exp_s), I_fc(R_plot, As_mean, Avs_mean, #wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp-R_exp_s), alpha=0.5, facecolor='y')
#plt.fill_between(R_plot, -I_fc(R_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp+R_exp_s), -I_fc(R_plot, As_mean, #Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp-R_exp_s), alpha=0.5, facecolor='y')
#
#### Plotte den coolen Kreis über I
#
#
#
#
#I_plot = np.linspace(I_min, I_max, 100000)
##R_min_tst = - Avs_mean / (2*As_mean) - np.sqrt( Avs_mean**2/(4*As_mean**2) + (R_exp-R_exp_s)/(2*As_mean) * ( wq_e_mean + wq_mu_mean ) - #wq_tau_mean/As_mean )
##R_max_tst = - Avs_mean / (2*As_mean) + np.sqrt( Avs_mean**2/(4*As_mean**2) + (R_exp-R_exp_s)/(2*As_mean) * ( wq_e_mean + wq_mu_mean ) - #wq_tau_mean/As_mean )
#
##I_plot = np.linspace( -I_fc( R_min_tst, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp),I_fc( R_min_tst, As_mean, Avs_mean, #wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp), 100000 )
#plt.plot(R_fc_up(I_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp), I_plot,'r')
#plt.plot(R_fc_down(I_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp), I_plot, 'r')
#
## fehlerdinger
#
#plt.fill_betweenx(I_plot, R_fc_up(I_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp-R_exp_s), R_fc_up(I_plot, As_mean, #Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp+R_exp_s) , alpha=0.5, facecolor='y')
#plt.fill_betweenx(I_plot, R_fc_down(I_plot, As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp-R_exp_s), R_fc_down(I_plot, #As_mean, Avs_mean, wq_mu_mean, wq_e_mean, wq_tau_mean, R_exp+R_exp_s), alpha=0.5, facecolor='y')
#
#
#
#
#plt.ylabel(r'$\mathrm{Im}(C_{s2})$')
#plt.xlabel(r'$\mathrm{Re}(C_{s1})$')
#plt.legend(loc='best')
#plt.savefig('plot_wilson_1_' + str(N1) + str(N2) + '.pdf') #fancy
#plt.clf()
#
#print("As =", As_mean)
#print("Asv =", Avs_mean)
#print("As^ =", As_mean / wq_tau_mean, "+-", As_std / wq_tau_mean)
#print("Asv^ =", Avs_mean / wq_tau_mean, "+-", Avs_std / wq_tau_mean)
#print("wq(tau) =", wq_tau_mean)
#
