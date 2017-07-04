import numpy as np
from params import m_tau, m_e, m_mu, m_p, m_0, m_b, m_d

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.set_cmap('Set2')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = ['\\usepackage{siunitx}']
plt.axis('equal')
plt.rc('font',family='Latin Modern')

def w(qq):
    return (m_b**2 + m_d**2 - qq)/(2*m_b*m_d)

def z(qq):
    w_tmp = w(qq)
    return ( np.sqrt(w_tmp + 1) - np.sqrt(2) )/( np.sqrt(w_tmp + 1) + np.sqrt(2) )

def re_z(qq):
    w_tmp = w(qq)
    if w_tmp>-1:
        return z(qq)
    else:
        return (-w_tmp - 3)/(-w_tmp + 1)

def im_z(qq):
    w_tmp = w(qq)
    if w_tmp >= -1:
        return 0
    else:
        return (2*np.sqrt(2) * np.sqrt(-w_tmp-1))/(-w_tmp + 1)


qq_min = m_e**2
qq_max = (m_b - m_d)**2


qq_area_3 = np.linspace(qq_max,100, 1000000)
plt.plot(list(map(re_z, qq_area_3)), list(map(im_z, qq_area_3)), 'r')
qq_area_32 = np.linspace(100,1000000, 1000000)
plt.plot(list(map(re_z, qq_area_32)), list(map(im_z, qq_area_32)), 'r')
qq_area_33 = np.linspace(-1000000,qq_min, 1000)
plt.plot(list(map(re_z, qq_area_33)), list(map(im_z, qq_area_33)), 'r--')
qq_area_4 = np.linspace(qq_min,qq_max, 100)
plt.plot(list(map(re_z, qq_area_4)), list(map(im_z, qq_area_4)), 'g', label=r'Kinematisch erlaubter Bereich.')

plt.ylabel(r'$\mathrm{Im}(z)$')
plt.xlabel(r'$\mathrm{Re}(z)$')
plt.legend(loc='best')

#plt.annotate(r'$q^2 \to \infty$',
#            xy=(0.73, 0.53), xycoords='data',
#            xytext=(-65, 60), textcoords='offset points',
#            arrowprops=dict(arrowstyle="->",
#                            connectionstyle="arc3,rad=-0.2", color="k"))

plt.annotate(r'$q^2 \to \infty$',
            xy=(0.80, 0.3), xycoords='data',
            xytext=(-65, 60), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2", color="k"))


plt.tight_layout()
plt.savefig('plot_z_2.pdf')
plt.clf()


qq_area = np.linspace(0, 13, 100000)
plt.plot(qq_area, w(qq_area), color='r')

plt.axvspan(qq_min, qq_max, alpha=0.2, color='blue', label=r'Kinematisch erlaubter Bereich')
plt.axhline(y=w(qq_min), linewidth=1, color='g', linestyle='dashed')#, label=r'$w(q^2_\text{min})$')
plt.axhline(y=w(qq_max), linewidth=1, color='y', linestyle='dashed')#, label=r'$w(q^2_\text{max})$')

plt.xlabel(r'$q^2 \,/\, \si{\giga\electronvolt\squared}$')
plt.legend(loc='best')
plt.ylabel(r'$w$')
plt.tight_layout()
plt.savefig('plot_w.pdf')
plt.clf()

qq_area_2 = np.linspace(0, 13, 100000)
plt.plot(qq_area_2, z(qq_area_2), color='r')

plt.axvspan(qq_min, qq_max, alpha=0.2, color='blue', label=r'Kinematisch erlaubter Bereich')
plt.axhline(y=z(qq_min), linewidth=1, color='g', linestyle='dashed')#, label=r'$z(q^2_\text{min})$')
plt.axhline(y=z(qq_max), linewidth=1, color='y', linestyle='dashed')#, label=r'$z(q^2_\text{max})$')

plt.xlabel(r'$q^2 \,/\, \si{\giga\electronvolt\squared}$')
plt.legend(loc='best')
plt.ylabel(r'$z$')
plt.tight_layout()
plt.savefig('plot_z.pdf')
plt.clf()
