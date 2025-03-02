import numpy as np
import openmdao.api as om
import math
from scipy import interpolate
import generatorse.wec_performance_module as wec_p

class Power(om.ExplicitComponent):
    def setup(self):
        self.add_input("T_e", 0.0, units="N*m", desc="Shear stress actual")
        self.add_input('gear_ratio', 0.0, units="m**-1", desc="Effective gear ratio")
        self.add_input('B_g', 0.0, units="T", desc="Peak air gap flux density B_g")
        self.add_input('mass_copper', 0.0, units="kg", desc="Copper mass")
        self.add_input('D_outer', 1.0, units='m', desc="Stator outer radius")
        self.add_input('l_s', 0.2, units='m', desc="Stack length")
        self.add_input('h_s', 0.01, units='m', desc="stator slot height")
        self.add_input('h_ys', 0.0, units='m', desc="stator yoke height")
        self.add_input('b_t', 0.01, units='m', desc="tooth width")
        self.add_input('N_s', 0.0, desc="Number of turns in the stator winding")
        self.add_input('rho_copper', units="kg/m**3", desc="Copper density kg/m^3")
        self.add_input('p', 0.0, desc="pole pairs")
        self.add_input('E_p', 0.0, units="V", desc="Stator phase voltage")
        self.add_input('m', 3, desc="number of phases")
        self.add_input('S', desc="Stator slots")
        self.add_input('r_inner', 0.0, units="m", desc="Inner radius of stator")
        self.add_input('h_s2', 0.010, units="m", desc="Slot wedge height")
        self.add_input('k_wd', desc="Winding factor")
        self.add_input("N_c", 0.0, desc="Number of turns per coil in series")

        self.add_output('P_max', 0.0, units="kW", desc="maximum power based on hydrodynamic simulation")
        self.add_output('I_max', 0.0, units="A", desc="Max stator current based on thermal constraints")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mass_copper = float(inputs["mass_copper"])
        r_outer = float(inputs["D_outer"])/2
        l_s = float(inputs["l_s"])
        h_s = float(inputs["h_s"])
        # h_ys = float(inputs["h_ys"])
        b_t = float(inputs["b_t"])
        N_s = float(inputs["N_s"])
        rho_copper = float(inputs["rho_copper"])
        p = float(inputs["p"])
        E_p = float(inputs["E_p"])
        m = float(inputs["m"])
        S = float(inputs["S"])
        T_e = float(inputs["T_e"])
        r_inner = float(inputs["r_inner"])
        h_s2 = float(inputs["h_s2"])
        N_c = float(inputs["N_c"])
        h_coef = 100.0    # Heat transfer coefficient (W/(m^2 degC))
        K_wb = 0.5      # Bare wire slot fill factor (-)
        t_pulse = 5     # time which peak torque can be sustained (s)
        T_amb = 30.0      # ambient/coolant temperature [degC]
        T_max = 120.0     # max allowable winding temp [decC]
        T_start = 50.0    # starting temp before a peak pulse (degC)
        c_copper = 377.0      # 
        τ = mass_copper*c_copper/(h_coef*2*np.pi*r_outer*l_s)    # thermal time constant of windings
        T_ss_pk = T_start+(T_max-T_start)/(1-math.exp(-t_pulse/τ))
        # R_sb = r_outer-h_ys        # radius of slot, toward the back of motor
        # d_sht = 0.00254       # distance from shoe to toe, assumed to 0.1 inch
        # R_out = 0.5 
        # R_sc = r_g + h_s1 + h_s2    # radius to coil
        B_c = 1e6   # coefficient of damping in powertrain
        K_c = 0.0     # coefficient of stiffness in powertrain
        wavefreq = 0.3      # frequency of waves as set in the sea state sim
        s = wavefreq*1j     # laplace variable, look at sea state stuff !!!!
        gear_ratio = float(inputs["gear_ratio"])       # need to incorperate gear ratio !!!!!
        B_g = float(inputs["B_g"])
        A_s = math.pi/S*(r_inner**2-(r_inner-h_s)**2)-b_t*(h_s2+h_s)   #area of single slot, from Hanselman eq 9.12
        K_t = 2*S/m*N_c*B_g*l_s*r_outer        # torque constant
        lambda_bar = 4*S/m*N_c*B_g*l_s*r_outer        # flux linkage
        # print("T_ss_pk: ",T_ss_pk)
        # print("A_s: ", A_s)
        I_max = (m**2/(S*N_c))*np.sqrt(np.pi*r_outer*h_coef*(T_ss_pk-T_amb)/(S*rho_copper/(A_s*K_wb)))
        print("I_max:", I_max)
        V_s = np.sqrt((B_c+K_c/s)**2+E_p**2)
        V_s_max_csv = V_s/(lambda_bar*p*gear_ratio)
        f_max_csv = K_t*I_max*gear_ratio
        f_max_csv2 = T_e*gear_ratio
        print("f_max_csv: ", f_max_csv)
        print("f_max_csv2: ", f_max_csv2)
        assert not np.isnan(V_s_max_csv), "V_s_max_csv is NaN!"
        assert not np.isnan(f_max_csv), "f_max_csv is NaN!"
        interp_power_sur = interpolate.RegularGridInterpolator((wec_p.force_u, wec_p.voltage_u), wec_p.power_surrogate)
        print("Grid Points:", interp_power_sur.grid)  # Tuple of arrays defining the grid
        print("Data Values:", interp_power_sur.values)  # The values at the grid points
        print("V_s_max_csv:", V_s_max_csv)
        print("f_max_csv", f_max_csv)
        P_max = interp_power_sur([V_s_max_csv, f_max_csv])/1000     # in kW
        print("P_max: ", P_max)
        print("I_max: ", I_max)
        outputs["I_max"] = I_max
        outputs["P_max"] = P_max

