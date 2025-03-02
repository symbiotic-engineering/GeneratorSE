import numpy as np
import openmdao.api as om
# import sys
# sys.path.append(r"C:\Users\Noam\generatorSE\GeneratorSE\ms_pmsg")
# from femm_fea import FEMM_Geometry


class Generator_Cost(om.ExplicitComponent):
    def setup(self):
        self.add_input("C_Cu", 0.0, units="USD/kg", desc="Specific cost of copper")
        self.add_input("C_Fe", 0.0, units="USD/kg", desc="Specific cost of magnetic steel/iron")
        self.add_input("C_Fes", 0.0, units="USD/kg", desc="Specific cost of structural steel")
        self.add_input("C_PM", 0.0, units="USD/kg", desc="Specific cost of magnet")
        self.add_input("C_NbTi", 0.0, units="USD/kg", desc="Specific cost of superconductor")

        # Mass of each material type
        self.add_input("mass_copper", 0.0, units="kg", desc="Copper mass")
        self.add_input("mass_iron", 0.0, units="kg", desc="Iron mass")
        self.add_input("mass_PM", 0.0, units="kg", desc="Magnet mass")
        self.add_input("mass_NbTi", 0.0, units="kg", desc="Superconductor mass")
        self.add_input("mass_structural", 0.0, units="kg", desc="Structural mass")

        self.add_input("D_generator", 0.0, units="m", desc="Armature diameter")
        self.add_input("hvac_mass_coeff", 0.025, units="kg/kW/m")
        self.add_input("hvac_mass_cost_coeff", 124.0, units="USD/kg")
        self.add_input("P_rated", units="kW", desc="Machine rating")
        
        self.add_input("mass_adder", 0.0, units="kg", desc="Mass to add to total for unaccounted elements")
        self.add_input("cost_adder", 0.0, units="USD", desc="Cost to add to total for unaccounted elements")

        # Outputs
        self.add_output("mass_active", 0.0, units="kg", desc="Total active mass")
        self.add_output("mass_hvac", 0.0, units="kg", desc="Cooling system mass")
        self.add_output("cost_hvac", 0.0, units="USD", desc="Cooling system cost")
        self.add_output("mass_total", 0.0, units="kg", desc="Total mass")
        self.add_output("cost_total", 0.0, units="USD", desc="Total cost")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Unpack inputs
        m_coeff = float(inputs["hvac_mass_coeff"])
        c_coeff = float(inputs["hvac_mass_cost_coeff"])
        rating = float(inputs["P_rated"])
        D_gen = float(inputs["D_generator"])
        m_copper = float(inputs["mass_copper"])
        m_iron = float(inputs["mass_iron"])
        m_pm = float(inputs["mass_PM"])
        m_sc = float(inputs["mass_NbTi"])
        m_struct = float(inputs["mass_structural"])
        m_add = float(inputs["mass_adder"])
        
        outputs["mass_hvac"] = m_hvac = m_coeff * rating * np.pi * D_gen
        outputs["cost_hvac"] = c_hvac = c_coeff * m_hvac

        # Assemble all masses for summation (either m_pm or m_sc will be zero)
        outputs["mass_active"] = m_active = m_copper + m_iron + m_pm + m_sc
        outputs["mass_total"] = m_active + m_struct + m_add + m_hvac

        # Compute material cost, with "buy-to-fly" ratios
        K_m = (1.26 * m_copper * inputs["C_Cu"] +
               1.21 * m_iron * inputs["C_Fe"] +
               1.00 * m_pm * inputs["C_PM"] +
               1.00 * m_sc * inputs["C_NbTi"] +
               1.21 * m_struct * inputs["C_Fes"])

        # Electricity usage, with kWh intensities from NREL model
        c_elec = 0.078 # 2022 $/kWh from EIA
        K_e = c_elec * (96.2 * m_copper + 26.9 * m_iron + 15.9 * m_struct +
                        79.0 * (m_sc + m_pm) )

        # Assemble all costs for now
        tempSum = K_m + K_e + inputs["cost_adder"] + c_hvac

        # Account for capital and labor cost share 18.9% capital, 61.9% materials, 19.3% labor
        outputs["cost_total"] = tempSum / 0.619
        
class LCOE_Calc(om.ExplicitComponent):
    def setup(self):
        self.add_input("cost_total", 0.0, units="USD", desc="Total cost")
        self.add_input("P_max", 0.0, units="kW", desc="Maximum power based on hydrodynamic simulation")

        self.add_output("LCOE", 0.0, units="USD/(kW*h)", desc="Levelized Cost Of Energy")
        self.add_output("AEP", 0.0, units="kW*h", desc="Annual Energy Production")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        cost_total = inputs["cost_total"]
        P_max = inputs["P_max"]
        # RM3 capital cost
        capex_wec = 518426277-2664921       # total capex based on sandia 2014 report adjusted for inlfation minus their generator cost
        # RM3 operational cost
        opex_wec = 12447258                 # total opex
        # Generator operational cost

        # Total Cost
        FCR = 0.113         # fixed change rate
        cost_ = (capex_wec+cost_total)*FCR+opex_wec
    
        # Energy generated in a year
        cap_fac = 0.3       # capacity factor of RM3
        AEP = P_max*cap_fac*8760    # annual energy production [kWh]
        outputs["AEP"] = AEP
        #print("Pmax:", P_max)
        #print("AEP:", AEP)
        outputs["LCOE"] = cost_/AEP         # levelized cost of energy