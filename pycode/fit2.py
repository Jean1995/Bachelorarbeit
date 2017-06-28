###
### mit MC Fehlerrechnung
### mit Korrelationen untereinander und Korellationsmatrix
### mit Korrigierten Funktionen
### mit verschiedenen Resonanzen
### mit scharfer Sauce

import numpy as np
from numpy.linalg import inv
import scipy.constants as const
from numpy import random
import uncertainties.unumpy as unp
from uncertainties import correlated_values, covariance_matrix, correlation_matrix
from table import (
    make_table,
    make_SI,
    write,
)

import locale
locale.setlocale(locale.LC_NUMERIC, "de_DE.utf8")

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.set_cmap('Set2')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.use_locale'] = True # kommata
plt.rcParams['text.latex.preamble'] = ['\\usepackage[locale=DE,separate-uncertainty=true,per-mode=symbol-or-fraction,]{siunitx} \\DeclareMathSymbol{,}{\mathord}{letters}{"3B}']
plt.rc('font',family='Latin Modern')

#plt.style.use('ggplot')
#plt.set_cmap('Set2')

import params
from params import N1, N2, m_p, m_0, m_b, m_d, m_e, m_tau, m_mu, m_bottom, m_charm, V_cb, R_exp, m_p_s, m_0_s, m_b_s, m_d_s, m_e_s, m_tau_s, m_mu_s, m_bottom_s, m_charm_s, V_cb_s, R_exp_s, corr_mat

m_array   = np.array([m_p, m_0, m_b, m_d, m_e, m_tau, m_mu, m_bottom, m_charm, V_cb, R_exp])
m_array_s = np.array([m_p_s, m_0_s, m_b_s, m_d_s, m_e_s, m_tau_s, m_mu_s, m_bottom_s, m_charm_s, V_cb_s, R_exp_s])

x_roh = params.w_roh
y = params.lattice_roh
s_y = params.s_l
V = params.V



### Funktionen

def z(w):
    return (np.sqrt(w+1) - np.sqrt(2)) / (np.sqrt(w+1) + np.sqrt(2))

def qq(z):
    return m_b**2 + m_d**2 - 2*m_b*m_d * (z**2 + 6*z + 1)/(z-1)**2

def z_from_qq(qq):
    w = (m_b**2 + m_d**2 - qq)/(2*m_b*m_d)
    return z(w)

z_max = z_from_qq(0)

def f(z,n,m):
    return z**n/(1 - qq(z)/m**2)

def f_s(z,n):
    if n==0:
        return 0
    else:
        return ( f(z, n, m_p) - f(z, 0, m_p)/f(z_max, 0, m_p) * f(z_max, n, m_p) )

def f_ss(z,n):
    return f(z_max, n, m_0)/f(z_max, 0, m_p) * f(z, 0, m_p)


def a00(a_tmp):
    '''Erhalte die 2*N-1 Parameter, wobei der Parameter a00 vorne fehlt. Dieser soll aus den restlichen errechnet werden'''
    a_00 = 0
    for i in range(N2):
        a_00 = a_00 + a_tmp[i+N1-1] * f(z_max, i, m_0)
    for i in range(N1-1):
        a_00 = a_00 - a_tmp[i] * f(z_max, i+1, m_p)
    a_00 = a_00 / f(z_max, 0, m_p)
    return a_00


### Daten


x = z(x_roh) # Werte für z

### Designmatrix
A_diag_1 = np.zeros((len(x), N1)) # Bezeichnungen siehe Blatt: Oben links
A_diag_2 = np.zeros((len(x), N2)) # Oben rechts
A_diag_3 = np.zeros((len(x), N1)) # Unten links (=Nullmatrix)
A_diag_4 = np.zeros((len(x), N2)) # Unten rechts
for i in range(N1):
    A_diag_1[:, i] = f_s(x,i)


for i in range(N2):
    A_diag_2[:, i] = f_ss(x,i)
    A_diag_4[:, i] = f(x, i, m_0)

A = np.vstack( (np.column_stack((A_diag_1, A_diag_2)) , np.column_stack((A_diag_3, A_diag_4)) ) ) # große Designmatrix
A = np.delete(A, 0, 1) # Workaround: Erste Spalte per Defintion null, da a_+0 = 0 per Defintion (wenn durch andere Funktionen ausgedrückt). Füge fehlenden Parameter am Ende hinzu

print("Designmatrix f+:")
print(A)
### Gewichtsmatrix

W = inv(V)

### Berechnung der Parameter

a_1 = inv(np.matmul(np.matmul(A.T, W), A))
a_2 = np.dot(np.matmul(A.T, W), y)
a = np.dot(a_1, a_2) # sollten N1 + N2 - 1 Werte sein.
a = np.insert(a, 0, a00(a)) # Berechne a_+0 aus den gegebenen Parametern

### Berechnung der Kovarianzmatrix der Parameter

V = a_1

### Nominelle Werte der Parameter

print("Parameter")
print(a)

### Plotten

x_plot = np.linspace(z_from_qq(0), z_from_qq((m_b-m_d)**2), 1000)

### Mr. Monte Carlo
samples = 1000
write('samples.tex', make_SI(samples, r'', figures=0))
x_plot_p_up = np.zeros(len(x_plot))
x_plot_p_down = np.zeros(len(x_plot))
x_plot_n_up = np.zeros(len(x_plot))
x_plot_n_down = np.zeros(len(x_plot))


a_mc = random.multivariate_normal(np.delete(a,0), V, samples)

#erstelle zufällige fehlerbehaftete Werte für Massen, Konstanten, etc.
# Erstellung m_array und fehler siehe oben
m_cov = np.diag(m_array_s)**2 # erstelle Kovarianzmatrix, jedoch ohne Kovarianzen (d.h. Varianzen auf Diagonalen)
m_mc = random.multivariate_normal(m_array, m_cov, samples) # m_mc[a,b] mit b bestimmt ich, welche Variable ich erhalte (also b=0 m_p, usw.)

a_00_vals = np.array([])
for i in range(samples):
    a_00_vals = np.insert(a_00_vals, 0, a00(a_mc[i,:]) ) # Ersetelle in array mit a_+0 werten für alle samples

a_mc = np.insert(a_mc, 0, a_00_vals, axis=1) # füge diese Spalte in die MonteCarlo-Daten ein

### Werte ausgeben


a_print = unp.uarray(a, np.insert(np.diag(V),1,np.std(a_00_vals))
) # füge den Fehler von a00 aus Monte-Carlo ein. Ya

tab_label = []
for i in range(N1):
    tab_label.append('a^+_' + str(i) + ': &')
for i in range(N2):
    tab_label.append('a^0_' + str(i) + ': &')

write('params_' + str(N1) + str(N2) + '.tex', make_table([a_print],[1]))


for i, item in enumerate(a_print):
    write('params_' + str(N1) + str(N2) + '_' + str(i)  + 'val.tex', make_table([[item]], [1]))
    np.savetxt('params_' + str(N1) + str(N2) + '_' + str(i)  + 'label.tex', [tab_label[i]], fmt='%s')





## Weiter MonteCarlo



for i, val in enumerate(x_plot):
    mc_values_p = np.zeros(samples) # für jedes z "sample"-viele Werte ermitteln
    mc_values_n = np.zeros(samples)
    for k in range(N1):
        mc_values_p = mc_values_p + a_mc.T[k]*f(val, k, m_mc[:,0]) # bestimme die Werte für f+, f0 für das jeweilige q^2
    for k in range(N2):
        mc_values_n = mc_values_n + a_mc.T[k+N1]*f(val, k, m_mc[:,1])
    mc_mean_p = np.mean(mc_values_p)
    mc_mean_n = np.mean(mc_values_n)
    mc_std_p = np.std(mc_values_p)
    mc_std_n = np.std(mc_values_n)
    x_plot_p_up[i] = mc_mean_p + 1*mc_std_p
    x_plot_p_down[i] = mc_mean_p - 1*mc_std_p
    x_plot_n_up[i] = mc_mean_n + 1*mc_std_n
    x_plot_n_down[i] = mc_mean_n - 1*mc_std_n

#
#### calculate chi-squared
#f+
pred_lat_val_fp = np.zeros(len(x))
for i in range(N1):
    pred_lat_val_fp = pred_lat_val_fp + a[i] * f(x, i, m_p)
chi_squared_fp = np.sum((( pred_lat_val_fp - np.split(y,2)[0] ) / (np.split(s_y,2)[0]))**2)



#fn
pred_lat_val_fn = np.zeros(len(x))
for i in range(N2):
    pred_lat_val_fn = pred_lat_val_fn + a[i+N1] * f(x, i, m_0)
chi_squared_fn = np.sum((( pred_lat_val_fn - np.split(y,2)[1] ) / (np.split(s_y,2)[1]))**2)

write('chisquared_p_' + str(N1) + str(N2) + '.tex', make_SI(chi_squared_fp, r'', figures=3))
write('chisquared_n_' + str(N1) + str(N2) + '.tex', make_SI(chi_squared_fn, r'', figures=3))
write('chisquared_sum_' + str(N1) + str(N2) + '.tex', make_SI(chi_squared_fp+chi_squared_fn, r'', figures=3))


# Plot f+

y_plot_p = np.zeros(len(x_plot))
for i in range(N1):
    y_plot_p = y_plot_p + a[i] * f(x_plot, i, m_p)

plt.plot(x_plot,y_plot_p, label=r'Fit $f_+$ mit Paramterzahl $N_+ = ' + str(N1) + r'$.', color='r')

plt.fill_between(x_plot, x_plot_p_up,  x_plot_p_down, interpolate=True, alpha=0.3, color='r',linewidth=0.0)

plt.errorbar( x, np.split(y,2)[0], yerr = np.split(s_y,2)[0], fmt=',', color='g', label=r'Theoriewerte $f_+$.', capsize=5,capthick=0.5, barsabove = True) # splitte in 2 Hälften und nehme die erste Hälfte

#plt.ylabel(r'$f_+(z)$')
#plt.xlabel(r'$z$')
#plt.legend(loc='best')
##plt.title(r'Fit der Formfaktoren. $\chi^2 = \num{' + str(chi_squared_fp) + r'}$.')
#plt.tight_layout()
#plt.savefig('plot_f+_' + str(N1) + str(N2) + '.pdf') #fancy
#plt.clf()




# Plot f0


y_plot_n = np.zeros(len(x_plot))
for i in range(N2):
    y_plot_n = y_plot_n + a[i+N1] * f(x_plot, i, m_0)

plt.plot(x_plot,y_plot_n, label=r'Fit $f_0$ mit Paramterzahl $N_0 = ' + str(N2) + r'$.', color='b')

plt.fill_between(x_plot, x_plot_n_up,  x_plot_n_down, interpolate=True, alpha=0.3, color='b', linewidth=0.0)

plt.errorbar( x, np.split(y,2)[1], yerr = np.split(s_y,2)[1], fmt=',', label=r'Theoriewerte $f_0$.', capsize=5,capthick=0.5, barsabove = True, color='y') # splitte in 2 Hälften und nehme die erste Hälfte

plt.ylabel(r'$f_i(z)$')
plt.xlabel(r'$z$')
plt.legend(loc='best')
#plt.title(r'Fit der Formfaktoren. $\chi^2 = \num{' + str(chi_squared_fn) + r'}$.')
plt.tight_layout()
plt.savefig('plot_' + str(N1) + str(N2) + '.pdf') #fancy
plt.clf()



### Teste: fp(0) =? fn(0)

print("f+(qq=0): ", y_plot_p[0])
print("f0+(qq=0): ", y_plot_n[0])

### Kovarianzmatrix -> Korellationsmatrix
x_label = []
for i in range(N1):
    if i > 0:
        x_label.append('$a^+_' + str(i) + '$')
for i in range(N2):
    x_label.append('$a^0_' + str(i) + '$')


D_inv = inv(np.sqrt(np.diag(np.diag(V))))
cor = np.matmul(D_inv, np.matmul(V, D_inv))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cor, interpolation='nearest', cmap='seismic')
cb = fig.colorbar(cax)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
cax.set_clim(vmin=-1, vmax=1)
#plt.title(r'Korrelationsmatrix')

for (i, j), z in np.ndenumerate(cor):
    plt.text(j, i, r'$\num{'+str('{:0.2f}'.format(z))+r'}$', ha='center', va='center', fontsize=15,
            bbox=dict(boxstyle='round', facecolor='grey', edgecolor='0.3'))

ax.set_xticklabels(['']+x_label, fontsize=15)
ax.set_yticklabels(['']+x_label, fontsize=15)

plt.tight_layout()
plt.savefig('cormatrix_a_N' + str(N1) + str(N2) + '.pdf')
plt.clf()

### Korellationsmatrix Daten

cor = corr_mat

x_label = []
for i in range(len(x_roh)):
    x_label.append(r'$f_+(\num{' + str(x_roh[i]) + r'})$')
for i in range(len(x_roh)):
    x_label.append(r'$f_0(\num{' + str(x_roh[i]) + r'})$')


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cor, interpolation='nearest', cmap='seismic')
#plt.colorbar(cax)
cb = fig.colorbar(cax)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
#cax.set_clim(vmin=-1, vmax=1)
#plt.title(r'Korrelationsmatrix Gittereichrechnungen.')

for (i, j), z in np.ndenumerate(cor):
    plt.text(j, i, r'$\num{'+str('{:0.2f}'.format(z))+r'}$', ha='center', va='center', fontsize=15,
            bbox=dict(boxstyle='round', facecolor='grey', edgecolor='0.3'))

ax.set_xticklabels(['']+x_label, fontsize=15)
ax.set_yticklabels(['']+x_label, fontsize=15)

plt.tight_layout()
plt.savefig('cormatrix_daten.pdf')
plt.clf()
