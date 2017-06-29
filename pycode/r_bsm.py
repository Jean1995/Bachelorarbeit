import numpy as np
from scipy.integrate import quad

from params import m_tau, N1, N2, eta, G_f, V_cb, m_e, m_mu, m_p, m_0, m_b, m_d, plot_difwq, R_exp, R_exp_s
from fit2 import a # Importiere die Parameter
from scipy.optimize import bisect

from table import (
    make_SI,
    write,
    make_table,
)

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


##### Funktionen

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


def f_p_new(z, a, alpha):
    '''
        Zu beachten: Die Reihenfolge der Parameter ist (a_+0, a+1, a+2, ... a+N-1, a_00, a01, a02, ... a0N-1)
    '''
    tmp = 0
    for n in range(N1):
        tmp = tmp + a[n] * f(z, n, m_p)
    return alpha * tmp

def f_0_new(z, a, beta):
    ''' '''
    tmp = 0
    for n in range(N2):
        tmp = tmp + a[n+N1] * f(z, n, m_0)
    return beta * tmp

def dif_wq_new(qq, m_l, a, alpha, beta):
    #beachte die gekürzten Faktoren: (eta^2 G_f^2 V_cb m_b)/(192 pi^3)
    r = m_d/m_b
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    c_plus = lambd/m_b**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    return np.sqrt(lambd) * (1 - m_l**2/qq)**2 * (c_plus * f_p_new(z_from_qq(qq), a, alpha)**2 + c_null * f_0_new(z_from_qq(qq), a, beta)**2)

def dif_wq_complete(qq, m_l, a, alpha, beta):
    r = m_d/m_b
    lambd = (qq - m_b**2 - m_d**2)**2 - 4*m_b**2*m_d**2
    c_plus = lambd/m_b**4 * (1 + m_l**2/(2*qq))
    c_null = (1-r**2)**2 * 3*m_l**2/(2*qq)
    vorfaktor = eta**2 * G_f**2 * V_cb**2 * m_b * np.sqrt(lambd) * 1/(192*np.pi**3) * (1 - m_l**2 / qq)**2
    return vorfaktor * (c_plus * alpha * f_p_new(z_from_qq(qq), a, alpha)**2 + c_null * beta * f_0_new(z_from_qq(qq), a, beta)**2 )


def f_R_exp(val_b, val_a, offset):
    tot_wq_e = quad(dif_wq_new, m_e**2, (m_b-m_d)**2, args=(m_e, a, val_a, val_b))[0]
    tot_wq_tau = quad(dif_wq_new, m_tau**2, (m_b-m_d)**2, args=(m_tau, a, val_a, val_b))[0]
    tot_wq_mu = quad(dif_wq_new, m_mu**2, (m_b-m_d)**2, args=(m_mu, a, val_a, val_b))[0]
    R_tmp = 2*tot_wq_tau/(tot_wq_e + tot_wq_mu)
    return R_tmp - (R_exp + offset)


### Bestimme Vorfaktoren vor Formfaktoren

alpha_min = 0.8
alpha_max = 1.2
num_alpha = 2
beta_min = 0.7
beta_max = 2.0 # wenn es einen ValueError gibt: beta_max höher setzen

### Zusatzbestimmung Alpha = 1

beta_val_nul = bisect(f_R_exp, beta_min, beta_max, args=(1, 0)) # suche das beta zum alpha welches R_exp ergibt
beta_val_up_nul = bisect(f_R_exp, beta_min, beta_max, args=(1, R_exp_s))
beta_val_down_nul = bisect(f_R_exp, beta_min, beta_max, args=(1, -R_exp_s))

good_values = []
good_values_up = []
good_values_down = []

### Allgemeine Bestimmung

for i, alpha in enumerate(np.linspace(alpha_min, alpha_max, num_alpha)):
    print("Progress calculating alpha/beta {:2.1%}".format(i / num_alpha), end="\r") #Kolleksche Bar
    beta_val = bisect(f_R_exp, beta_min, beta_max, args=(alpha, 0)) # suche das beta zum alpha welches R_exp ergibt
    good_values.append((alpha, beta_val))
    beta_val_up = bisect(f_R_exp, beta_val, beta_max, args=(alpha, R_exp_s))
    good_values_up.append((alpha, beta_val_up))
    beta_val_down = bisect(f_R_exp, beta_min, beta_val, args=(alpha, -R_exp_s))
    good_values_down.append((alpha, beta_val_down))
    beta_min = beta_val_down


good_values = np.array(good_values)
good_values_up = np.array(good_values_up)
good_values_down = np.array(good_values_down)


plt.plot(good_values[:,0], good_values[:,1], '--', label=r'Parameterbereich $(\alpha,\: \beta$) mit $ R_{\alpha, \beta} = R_{\text{exp}}$')
plt.fill_between(good_values_up[:,0], good_values_up[:,1],  good_values_down[:,1], interpolate=True, alpha=0.5, label=r'Parameterbereich $(\alpha,\: \beta)$ mit $ R_{\alpha, \beta} \in \left[ R_{\text{exp}} - \sigma, R_{\text{exp}} + \sigma \right] $')
plt.plot(1, beta_val_nul, 'g*',markersize=12, label=r'$ R = R_{\text{exp}}$ mit $\alpha = 1$ und $\beta = \num[round-mode=places,round-precision=2]{' + str(beta_val_nul) + r'}$')

write('beta_val_nul.tex', make_SI(beta_val_nul, r'', figures=2))


#plt.plot(1, beta_val_nul, 'x', label=r'$ R = R_{\text{exp}}$ mit $\alpha = 1$ und $\beta = \num[round-mode=places,round-precision=2]{' + str(beta_val_nul) + r'}$')
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\alpha$')
#plt.axvline(x=1, linewidth=1, color = 'g')
#plt.ylim(alpha_min, beta_max-0.1)
plt.xlim(alpha_min, alpha_max)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('alpha_beta_' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()



print("Parameter für alpha = 1:")
print("beta = " + str(beta_val_nul))
print("bzw von" + str(beta_val_down_nul) + " bis" + str(beta_val_up_nul))

### Plot für alpha = 1

red = 1/(10**(-15) )#* 10**9 * const.eV)

qq_plot_tau = np.linspace(m_tau**2, (m_b-m_d)**2 , 300)
qq_plot_e = np.linspace(m_e**2, (m_b-m_d)**2 , 300)
qq_plot_mu = np.linspace(m_mu**2, (m_b-m_d)**2 , 300)

plt.plot(z_from_qq(qq_plot_e) ,dif_wq_complete(qq_plot_e, m_e, a, 1, beta_val_nul)*red, 'b', label=r'Differentielle Zerfallsbreite mit $l = e$, $\beta = \num[round-mode=places,round-precision=2]{' + str(beta_val_nul) + r'}$')

plt.plot(z_from_qq(qq_plot_tau) ,dif_wq_complete(qq_plot_tau, m_tau, a, 1, beta_val_nul)*red, 'r', label=r'Differentielle Zerfallsbreite mit $l = \tau$, $\beta = \num[round-mode=places,round-precision=2]{' + str(beta_val_nul) + r'}$')

plt.plot(z_from_qq(qq_plot_mu) ,dif_wq_complete(qq_plot_mu, m_mu, a, 1, beta_val_nul)*red, 'g', label=r'Differentielle Zerfallsbreite mit $l = \mu$, $\beta = \num[round-mode=places,round-precision=2]{' + str(beta_val_nul) + r'}$')

plt.plot(z_from_qq(qq_plot_e) ,dif_wq_complete(qq_plot_e, m_e, a, 1, 1)*red, 'b--', label=r'Differentielle Zerfallsbreite mit $l = e$, $\beta = 1$')

plt.plot(z_from_qq(qq_plot_tau) ,dif_wq_complete(qq_plot_tau, m_tau, a, 1, 1)*red, 'r--', label=r'Differentielle Zerfallsbreite mit $l = \tau$, $\beta = 1$')

plt.plot(z_from_qq(qq_plot_mu) ,dif_wq_complete(qq_plot_mu, m_mu, a, 1, 1)*red, 'g--', label=r'Differentielle Zerfallsbreite mit $l = \mu$, $\beta = 1$')

plt.ylabel(r'$\frac{d \Gamma}{d q^2} \left(B \to D l \nu_l \right) \,/\, \left( \num{e-15} \si{\giga\electronvolt\tothe{-1}} \right)$')
plt.xlabel(r'$z$')
plt.legend(loc='best', prop={'size':17})
plt.tight_layout()
plt.savefig('plot_diff_wq_Rexp' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()
