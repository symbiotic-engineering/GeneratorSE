# import sys
# sys.path.append(r"C:\Users\Noam\generatorSE\GeneratorSE")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from scipy.optimize import curve_fit
# import WecOptTool results csv
df = pd.read_csv(r"C:\Users\Noam\generatorSE\GeneratorSE\generatorse\power_sensitivities_2.csv")

# extract the points where position is unconstrained
#max_x = df['x_max'].max()
#df_x_unconstrained = df[df['x_max'] == max_x]
df_x_unconstrained = df
force = df_x_unconstrained['f_max']
voltage = df_x_unconstrained['Vs_max']
power = -df_x_unconstrained['electrical Power']
#x_max = df_x_unconstrained['x_max']

# reshape to 2D
force_u = force.unique()
voltage_u = voltage.unique()
num_forces = len(force_u)
num_voltages = len(voltage_u)
force = force.values.reshape(num_forces, num_voltages)
voltage = voltage.values.reshape(num_forces, num_voltages)
power = power.values.reshape(num_forces, num_voltages)

# contour plot of power
plt.contourf(force/1000, voltage/1000, power/1000)
plt.colorbar(label='Power (kW)')
plt.scatter(force/1000, voltage/1000, c='k')
plt.xlabel('Force (kN)')
plt.ylabel('Voltage (kV)')

#print("power:\n", power)
P_max = np.max(power)
force_const = 280e3
voltage_const = 55e3
#Ftemp = list(np.linspace(100, 1000, 5))
#F = np.array([Ftemp, Ftemp, Ftemp, Ftemp, Ftemp]).flatten()
#Vtemp = list(np.linspace(0, 300, 5))
#V = np.array([Vtemp, Vtemp, Vtemp, Vtemp, Vtemp]).flatten()
#f, v = np.linspace(100, 1000, 5), np.linspace(0, 300, 5)
#F, V = np.meshgrid(f, v)
#def guess(f, v, af, av, bf, bv, fc, vc):
#    return P_max * (1 - np.exp(-(f/fc)**af)+bf) * (1 - np.exp(-(v/vc)**av)+bv)

def guess(f, v, a0, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3):
    return P_max * (a4*f**4+a3*f**3+a2*f**2+a1*f+
                    b4*v**4+b3*v**3+b2*v**2+b1*v+
                    c3*(f*v)**3+c2*(f*v)**2+c1*f*v+
                    a0)

def _guess(M, *args):
    f, v = M
    arr = guess(f, v, *args)
    #arr = np.zeros(f.shape)
    #for i in range(len(args)//6):
    #    arr += guess(f, v, *args[i*6:i*6+6])
    return arr

#xdata = np.vstack((F.ravel(), V.ravel()))
xdata = np.vstack((force.ravel(), voltage.ravel()))

p0 = [[-6.44573088e-02,  2.47574425e-06, -4.25647148e-12,  2.52700057e-18,
-4.40747842e-25,  3.68565273e-06, -3.81058251e-11,  1.32107747e-16,
-1.64089344e-22,  6.44846086e-12, -1.91233531e-23,  2.35070957e-35]]
# p0 = [1.5, 0.65, 0.075, 0.65, 280e3, 30e3]
# opt_param, cov = curve_fit(_guess, xdata, power.ravel(), p0=p0, bounds=((0, 0, 0, 0, 0.0001, 0.0001), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
opt_param, cov = curve_fit(_guess, xdata, power.ravel(), p0=p0, bounds=((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)))
#print('p0:\n', p0)
#print('optimization parameters:\n',opt_param)

power_surrogate = guess(force, voltage, *opt_param)
power_surrogate = power_surrogate.reshape(num_forces, num_voltages)
