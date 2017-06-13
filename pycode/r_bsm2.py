import numpy as np
from params import m_tau, N1, N2, eta, G_f, V_cb, m_e, m_mu, m_p, m_0, m_b, m_d, plot_difwq, R_exp, R_exp_s, m_bottom, m_charm
from scipy.integrate import quad

from fit2 import a_mc # Importiere die Monte Carlo Parameter
from fit2 import a as a_genau
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

def Hss(qq, a):
    return (m_b**2 - m_d**2)/(m_bottom - m_charm) * f_0(z_from_qq(qq), a)

def Hvts(qq, a):
    return (m_b**2 - m_d**2)/np.sqrt(qq) * f_0(z_from_qq(qq), a)

def Hv0s(qq, a):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return np.sqrt(lambd/qq) * f_p(z_from_qq(qq), a)


### BSM Funktionen aus Paper https://arxiv.org/pdf/1309.0301.pdf (8)

def A_s(qq, m_l, a):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3/2 * Hss(qq, a)**2

def A_vs(qq, m_l, a):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3 * m_l/np.sqrt(qq) * Hss(qq, a) * Hvts(qq, a)

def wq(qq, m_l, a):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * ( (1 + m_l**2 /(2*qq))* Hv0s(qq, a)**2 + 3/2 * m_l**2/qq * Hvts(qq, a)**2 )


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

for a in a_mc:
    tmp_as = quad(A_s, m_tau**2, (m_b-m_d)**2, args=(m_tau, a))[0]
    tmp_avs = quad(A_vs, m_tau**2, (m_b-m_d)**2, args=(m_tau, a))[0]
    tmp_wq_tau = quad(wq, m_tau**2, (m_b-m_d)**2, args=(m_tau, a))[0]
    tmp_wq_e = quad(wq, m_e**2, (m_b-m_d)**2, args=(m_e, a))[0]
    tmp_wq_mu = quad(wq, m_mu**2, (m_b-m_d)**2, args=(m_mu, a))[0]
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

As_mean = np.mean(As_values)
As_std = np.std(As_values)
Avs_mean = np.mean(Avs_values)
Avs_std = np.std(Avs_values)
wq_tau_mean = np.mean(wq_tau_values)
wq_e_mean = np.mean(wq_e_values)
wq_mu_mean = np.mean(wq_mu_values)

As_exact = quad(A_s, m_tau**2, (m_b-m_d)**2, args=(m_tau, a_genau))[0]
Avs_exact = quad(A_vs, m_tau**2, (m_b-m_d)**2, args=(m_tau, a_genau))[0]
wq_tau_exact = quad(wq, m_tau**2, (m_b-m_d)**2, args=(m_tau, a_genau))[0]
wq_e_exact = quad(wq, m_e**2, (m_b-m_d)**2, args=(m_e, a_genau))[0]
wq_mu_exact = quad(wq, m_mu**2, (m_b-m_d)**2, args=(m_mu, a_genau))[0]


### Plotte den coolen Kreis über R

R_min = - Avs_exact / (2*As_exact) - np.sqrt( Avs_exact**2/(4*As_exact**2) + R_exp/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )
R_max = - Avs_exact / (2*As_exact) + np.sqrt( Avs_exact**2/(4*As_exact**2) + R_exp/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )
I_min = - np.sqrt( Avs_exact**2/(4*As_exact**2) + R_exp/(2*As_exact) * (wq_e_exact + wq_mu_exact) - wq_tau_exact/As_exact )
I_max = + np.sqrt( Avs_exact**2/(4*As_exact**2) + R_exp/(2*As_exact) * (wq_e_exact + wq_mu_exact) - wq_tau_exact/As_exact )

R_plot = np.linspace(R_min, R_max, 100000)


#
## fehlerdinger
#
#plt.fill_between(R_plot, I_fc(R_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s), I_fc(R_plot, As_exact, Avs_exact, #wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s), alpha=0.5, facecolor='y')
#plt.fill_between(R_plot, -I_fc(R_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s), -I_fc(R_plot, As_exact, #Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s), alpha=0.5, facecolor='y')
#
#### Plotte den coolen Kreis über I
#
#
#
#
#I_plot = np.linspace(I_min, I_max, 100000)
##R_min_tst = - Avs_exact / (2*As_exact) - np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp-R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - #wq_tau_exact/As_exact )
##R_max_tst = - Avs_exact / (2*As_exact) + np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp-R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - #wq_tau_exact/As_exact )
#
##I_plot = np.linspace( -I_fc( R_min_tst, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp),I_fc( R_min_tst, As_exact, Avs_exact, #wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp), 100000 )
#plt.plot(R_fc_up(I_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp), I_plot,'r')
#plt.plot(R_fc_down(I_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp), I_plot, 'r')
#
# fehlerdinger

R_min_down = - Avs_exact / (2*As_exact) - np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp-R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )
R_max_down = - Avs_exact / (2*As_exact) + np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp-R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )

R_plot_down = np.linspace(R_min_down, R_max_down-0.0001, 100000)

R_min_up = - Avs_exact / (2*As_exact) - np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp+R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )
R_max_up = - Avs_exact / (2*As_exact) + np.sqrt( Avs_exact**2/(4*As_exact**2) + (R_exp+R_exp_s)/(2*As_exact) * ( wq_e_exact + wq_mu_exact ) - wq_tau_exact/As_exact )


R_plot_up = np.linspace(R_min_up, R_max_up-0.0001, 100000)

ys = np.array([ np.concatenate((I_fc(R_plot_up, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s), -I_fc(np.flipud(R_plot_up), As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s))) , np.concatenate((I_fc(R_plot_down, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s) ,-I_fc(np.flipud(R_plot_down), As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s))) ] )

xs = np.array( [ np.concatenate((R_plot_up, np.flipud(R_plot_up))) , np.concatenate((R_plot_down, np.flipud(R_plot_down))) ] )

#plt.fill_betweenx(I_plot, R_fc_up(I_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s), R_fc_up(I_plot, As_exact, #Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s) , alpha=0.5, facecolor='y')
#plt.fill_betweenx(I_plot, R_fc_down(I_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp-R_exp_s), R_fc_down(I_plot, #As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp+R_exp_s), alpha=0.5, facecolor='y')


xs[1,:] = xs[1,::-1]
ys[1,:] = ys[1,::-1]
ax = plt.subplot(111, aspect='equal')
ax.fill(np.ravel(xs), np.ravel(ys), alpha = 0.5, lw=0)

plt.plot(R_plot, I_fc(R_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp), 'r' )
plt.plot(R_plot, -I_fc(R_plot, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp), 'r' )

plt.ylabel(r'$\mathrm{Im}(C_{s2})$')
plt.xlabel(r'$\mathrm{Re}(C_{s1})$')
plt.legend(loc='best')
plt.savefig('plot_wilson_1_' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()

print("As =", As_mean)
print("Asv =", Avs_mean)
print("As^ =", As_mean / wq_tau_mean, "+-", As_std / wq_tau_mean)
print("Asv^ =", Avs_mean / wq_tau_mean, "+-", Avs_std / wq_tau_mean)
print("wq(tau) =", wq_tau_mean)
