import numpy as np
from params import m_tau, N1, N2, eta, G_f, V_cb, m_e, m_mu, m_p, m_0, m_b, m_d, plot_difwq, R_exp, R_exp_s, m_bottom, m_charm
from scipy.integrate import quad

from fit2 import a_mc # Importiere die Monte Carlo Parameter
from fit2 import m_mc
from fit2 import a as a_genau
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.set_cmap('Set2')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.use_locale'] = True # kommata
plt.rcParams['text.latex.preamble'] = ['\\usepackage[locale=DE,separate-uncertainty=true,per-mode=symbol-or-fraction,]{siunitx} \\DeclareMathSymbol{,}{\mathord}{letters}{"3B}']
plt.rc('font',family='Latin Modern')

from table import (
    make_table,
    make_SI,
    write,
)
from uncertainties import ufloat


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


def f_p(z, a, m_p = m_p):
    '''
        Zu beachten: Die Reihenfolge der Parameter ist (a_+0, a+1, a+2, ... a+N-1, a_00, a01, a02, ... a0N-1)
    '''
    tmp = 0
    for n in range(N1):
        tmp = tmp + a[n] * f(z, n, m_p)
    return tmp

def f_0(z, a, m_0 = m_0):
    ''' '''
    tmp = 0
    for n in range(N2):
        tmp = tmp + a[n+N1] * f(z, n, m_0)
    return tmp

def Hss(qq, a, m_b = m_b, m_d = m_d, m_bottom=m_bottom, m_charm=m_charm, m_0=m_0):
    return (m_b**2 - m_d**2)/(m_bottom - m_charm) * f_0(z_from_qq(qq), a, m_0)

def Hvts(qq, a, m_b=m_b, m_d=m_d, m_0=m_0):
    return (m_b**2 - m_d**2)/np.sqrt(qq) * f_0(z_from_qq(qq), a, m_0)

def Hv0s(qq, a, m_b=m_b, m_d=m_d, m_p=m_p):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return np.sqrt(lambd/qq) * f_p(z_from_qq(qq), a, m_p)


### BSM Funktionen aus Paper https://arxiv.org/pdf/1309.0301.pdf (8)

def A_s(qq, m_l, a, m_b=m_b, m_d=m_d, V_cb=V_cb, m_bottom=m_bottom, m_charm=m_charm, m_0=m_0):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3/2 * Hss(qq, a, m_b, m_d, m_bottom, m_charm, m_0)**2

def A_vs(qq, m_l, a, m_b=m_b, m_d=m_d, V_cb=V_cb, m_bottom=m_bottom, m_charm=m_charm, m_0=m_0):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * 3 * m_l/np.sqrt(qq) * Hss(qq, a, m_b, m_d, m_bottom, m_charm, m_0) * Hvts(qq, a, m_b, m_d, m_0)

def wq(qq, m_l, a, m_b=m_b, m_d=m_d, V_cb=V_cb, m_p=m_p):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * ( (1 + m_l**2 /(2*qq))* Hv0s(qq, a, m_b, m_d, m_p)**2 + 3/2 * m_l**2/qq * Hvts(qq, a, m_b, m_d, m_0)**2 )

def wq_bsm(qq, m_l, a, R_c, I_c, m_b=m_b, m_d=m_d, V_cb=V_cb, m_p=m_p, m_bottom=m_bottom, m_charm= m_charm, m_0= m_0):
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    return (eta * G_f**2 * V_cb**2)/(192 * np.pi**3 * m_b**3) * qq * np.sqrt(lambd) * (1 - m_l**2 / qq)**2 * ( (1 + m_l**2 /(2*qq))* Hv0s(qq, a, m_b, m_d, m_p)**2 + 3/2 * m_l**2/qq * Hvts(qq, a, m_b, m_d, m_0)**2  + 3/2 * (R_c**2 + I_c**2) * Hss(qq, a, m_b, m_d, m_bottom, m_charm, m_0)**2 + 3*R_c * m_l/np.sqrt(qq) * Hss(qq, a, m_b, m_d, m_bottom, m_charm, m_0) * Hvts(qq, a, m_b, m_d, m_0))

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

for a, m in zip(a_mc, m_mc):
    tmp_as = quad(A_s, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[7], m[8], m[1]))[0]
    tmp_avs = quad(A_vs, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[7], m[8], m[1]))[0]
    tmp_wq_tau = quad(wq, m[5]**2, (m[2]-m[3])**2, args=(m[5], a, m[2], m[3], m[9], m[0]))[0]
    tmp_wq_e = quad(wq, m[4]**2, (m[2]-m[3])**2, args=(m[4], a, m[2], m[3], m[9], m[0]))[0]
    tmp_wq_mu = quad(wq, m[6]**2, (m[2]-m[3])**2, args=(m[6], a, m[2], m[3], m[9], m[0]))[0]
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

# Fehlerdinger

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

#Plotte diese Punkte, da für diese Punkte gpelottet werden soll


I_plt_0 = 0
R_plt_0 = 0
I_plt_1 = 0
I_plt_2 = 0
R_plt_1 = R_fc_up(I_plt_1, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp)
R_plt_2 = R_fc_down(I_plt_2, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp)
I_plt_3 = I_max-0.000001
R_plt_3 = R_fc_up(I_plt_3, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp)
I_plt_4 = I_min+0.000001
R_plt_4 = R_fc_down(I_plt_4, As_exact, Avs_exact, wq_mu_exact, wq_e_exact, wq_tau_exact, R_exp)

plt.plot(R_plt_2, I_plt_2, 'b*',markersize=10, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_2) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_2) + r'}$')
plt.plot(R_plt_1, I_plt_1, 'y*',markersize=10, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_1) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_1) + r'}$')
plt.plot(R_plt_3, I_plt_3, 'g*',markersize=10, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_3) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_3) + r'}$')
plt.plot(R_plt_4, I_plt_4, 'm*',markersize=10, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_4) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_4) + r'}$')

plt.ylabel(r'$\mathrm{Im}(C_{\text{S}1})$')
plt.xlabel(r'$\mathrm{Re}(C_{\text{S}1})$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot_wilson_1_' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()


# Plotte mal die dif ZB tau

qq_plot_tau = np.linspace(m_tau**2, (m_b-m_d)**2, 300)
dif_wq_val_tau_0 = []
dif_wq_tau_up_0 = []
dif_wq_tau_down_0 = []
dif_wq_val_tau_1 = []
dif_wq_tau_up_1 = []
dif_wq_tau_down_1 = []
dif_wq_val_tau_2 = []
dif_wq_tau_up_2 = []
dif_wq_tau_down_2 = []
dif_wq_val_tau_3 = []
dif_wq_tau_up_3 = []
dif_wq_tau_down_3 = []
dif_wq_val_tau_4 = []
dif_wq_tau_up_4 = []
dif_wq_tau_down_4 = []

for qq_tmp in qq_plot_tau:
    tmp_tau_0 = []
    tmp_tau_1 = []
    tmp_tau_2 = []
    tmp_tau_3 = []
    tmp_tau_4 = []

    for a, m in zip(a_mc, m_mc):
        tmp_tau_0.append( wq_bsm(qq_tmp, m[5], a, R_plt_0, I_plt_0, m[2], m[3], m[9], m[0], m[7], m[8], m[1]) )
        tmp_tau_1.append( wq_bsm(qq_tmp, m[5], a, R_plt_1, I_plt_1, m[2], m[3], m[9], m[0], m[7], m[8], m[1]) )
        tmp_tau_2.append( wq_bsm(qq_tmp, m[5], a, R_plt_2, I_plt_2, m[2], m[3], m[9], m[0], m[7], m[8], m[1]) )
        tmp_tau_3.append( wq_bsm(qq_tmp, m[5], a, R_plt_3, I_plt_3, m[2], m[3], m[9], m[0], m[7], m[8], m[1]) )
        tmp_tau_4.append( wq_bsm(qq_tmp, m[5], a, R_plt_4, I_plt_4, m[2], m[3], m[9], m[0], m[7], m[8], m[1]) )
    tmp_mean_tau_0 = np.mean(tmp_tau_0)
    tmp_std_tau_0 = np.std(tmp_tau_0)
    dif_wq_val_tau_0.append(tmp_mean_tau_0)
    dif_wq_tau_up_0.append(tmp_mean_tau_0 + tmp_std_tau_0)
    dif_wq_tau_down_0.append(tmp_mean_tau_0 - tmp_std_tau_0)
    tmp_mean_tau_1 = np.mean(tmp_tau_1)
    tmp_std_tau_1 = np.std(tmp_tau_1)
    dif_wq_val_tau_1.append(tmp_mean_tau_1)
    dif_wq_tau_up_1.append(tmp_mean_tau_1 + tmp_std_tau_1)
    dif_wq_tau_down_1.append(tmp_mean_tau_1 - tmp_std_tau_1)
    tmp_mean_tau_2 = np.mean(tmp_tau_2)
    tmp_std_tau_2 = np.std(tmp_tau_2)
    dif_wq_val_tau_2.append(tmp_mean_tau_2)
    dif_wq_tau_up_2.append(tmp_mean_tau_2 + tmp_std_tau_2)
    dif_wq_tau_down_2.append(tmp_mean_tau_2 - tmp_std_tau_2)
    tmp_mean_tau_3 = np.mean(tmp_tau_3)
    tmp_std_tau_3 = np.std(tmp_tau_3)
    dif_wq_val_tau_3.append(tmp_mean_tau_3)
    dif_wq_tau_up_3.append(tmp_mean_tau_3 + tmp_std_tau_3)
    dif_wq_tau_down_3.append(tmp_mean_tau_3 - tmp_std_tau_3)
    tmp_mean_tau_4 = np.mean(tmp_tau_4)
    tmp_std_tau_4 = np.std(tmp_tau_4)
    dif_wq_val_tau_4.append(tmp_mean_tau_4)
    dif_wq_tau_up_4.append(tmp_mean_tau_4 + tmp_std_tau_4)
    dif_wq_tau_down_4.append(tmp_mean_tau_4 - tmp_std_tau_4)

dif_wq_val_tau_0 = np.array(dif_wq_val_tau_0)
dif_wq_tau_up_0 = np.array(dif_wq_tau_up_0)
dif_wq_tau_down_0 = np.array(dif_wq_tau_down_0)
dif_wq_val_tau_1 = np.array(dif_wq_val_tau_1)
dif_wq_tau_up_1 = np.array(dif_wq_tau_up_1)
dif_wq_tau_down_1 = np.array(dif_wq_tau_down_1)
dif_wq_val_tau_2 = np.array(dif_wq_val_tau_2)
dif_wq_tau_up_2 = np.array(dif_wq_tau_up_2)
dif_wq_tau_down_2 = np.array(dif_wq_tau_down_2)
dif_wq_val_tau_3 = np.array(dif_wq_val_tau_3)
dif_wq_tau_up_3 = np.array(dif_wq_tau_up_3)
dif_wq_tau_down_3 = np.array(dif_wq_tau_down_3)
dif_wq_val_tau_4 = np.array(dif_wq_val_tau_4)
dif_wq_tau_up_4 = np.array(dif_wq_tau_up_4)
dif_wq_tau_down_4 = np.array(dif_wq_tau_down_4)

red = 1/(10**(-15) )#* 10**9 * const.eV)


plt.plot(z_from_qq(qq_plot_tau), dif_wq_val_tau_2*red, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_2) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_2) + r'}$', color='b')
plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_tau_up_2*red, dif_wq_tau_down_2*red, interpolate=True, alpha=0.5, color='b')

plt.plot(z_from_qq(qq_plot_tau), dif_wq_val_tau_3*red, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_3) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \pm\num[round-mode=places,round-precision=2]{' + str(I_plt_3) + r'}$', color='g')
plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_tau_up_3*red, dif_wq_tau_down_3*red, interpolate=True, alpha=0.5, color='g')

plt.plot(z_from_qq(qq_plot_tau), dif_wq_val_tau_1*red, label=r'$\mathrm{Re}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_1) + r'}$ und $\mathrm{Im}(C_{\text{S}1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_1) + r'}$', color='y')
plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_tau_up_1*red, dif_wq_tau_down_1*red, interpolate=True, alpha=0.5, color='y')

plt.plot(z_from_qq(qq_plot_tau), dif_wq_val_tau_0*red, label=r'Standardmodell', color='r')
plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_tau_up_0*red, dif_wq_tau_down_0*red, interpolate=True, alpha=0.5, color='r')

write('Cs1_Re.tex', make_SI(R_plt_3, r'', figures=2))
write('Cs1_Im.tex', make_SI(I_plt_3, r'', figures=2))


#plt.plot(z_from_qq(qq_plot_tau), dif_wq_val_tau_4*red, label=r'$\mathrm{Re}(C_{s1}) = \num[round-mode=places,round-precision=2]{' + str(R_plt_4) + r'}$ und $\mathrm{Im}(C_{s1}) = \num[round-mode=places,round-precision=2]{' + str(I_plt_4) + r'}$', color='m')
#plt.fill_between(z_from_qq(qq_plot_tau), dif_wq_tau_up_4*red, dif_wq_tau_down_4*red, interpolate=True, alpha=0.5, color='m')



plt.xlabel(r'$z$')
plt.ylabel(r'$\frac{\mathrm{d} \Gamma}{\mathrm{d} q^2} \left(\overline{B} \to D \tau \overline{\nu}_{\tau} \right) \,/\, \left( \num{e-15} \si{\giga \electronvolt\tothe{-1}} \right)$')
plt.legend(loc='best', prop={'size':16})
plt.tight_layout()
plt.savefig('plot_bsm_dif_wq_' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()


print("As =", As_mean)
print("Asv =", Avs_mean)
print("As^ =", As_mean / wq_tau_mean, "+-", As_std / wq_tau_mean)
print("Asv^ =", Avs_mean / wq_tau_mean, "+-", Avs_std / wq_tau_mean)
print("wq(tau) =", wq_tau_mean)

write('As.tex', make_SI(ufloat(As_mean / wq_tau_mean,As_std / wq_tau_mean), r'', figures=1))
write('Avs.tex', make_SI(ufloat(Avs_mean / wq_tau_mean,Avs_std / wq_tau_mean), r'', figures=1))
