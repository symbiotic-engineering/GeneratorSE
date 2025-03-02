# import sys
# sys.path.append(r"C:\Users\Noam\generatorSE\GeneratorSE")
import openmdao.api as om
from generatorse.ms_pmsg.ms_pmsg import PMSG_Inner_rotor_Opt
from generatorse.ms_pmsg.structural import PMSG_Inner_Rotor_Structural
from generatorse.driver.nlopt_driver import NLoptDriver
from generatorse.common.femm_util import cleanup_femm_files
from generatorse.common.run_util import copy_data, load_data, save_data
#from generatorse.common.cost import Generator_Cost
import os
import pandas as pd
import numpy as np
import platform

if platform.system().lower() == 'darwin':
    os.environ["CX_BOTTLE"] = "FEMM"
    os.environ["WINEPATH"] = "/Users/gbarter/bin/wine"
    os.environ["FEMMPATH"] = "/Users/gbarter/Library/Application Support/CrossOver/Bottles/FEMM/drive_c/femm42/bin/femm.exe"

# gear_ratio = 120
# ratings_known = [15, 17, 20, 22, 25]
# rotor_diameter = {}
# rotor_diameter[15] = 242.24
# rotor_diameter[17] = 257.88
# rotor_diameter[20] = 279.71
# rotor_diameter[22] = 293.36
# rotor_diameter[25] = 312.73
rated_speed = {}
# rated_speed[15] = 7.49      #gear_ratio * 7.49
# rated_speed[17] = 7.04      #gear_ratio * 7.04
# rated_speed[20] = 6.49      #gear_ratio * 6.49
# rated_speed[22] = 6.18      #gear_ratio * 6.18
# rated_speed[25] = 5.80      #gear_ratio * 5.80
rated_speed[286] = 8.0      # guess
target_eff = 0.97
fsql = "log.sql"
output_root = "MS-PMSG_output"
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

def validate_inputs(prob):
    for var_name, meta in prob.model._design_vars.items():
        lower = meta['lower']
        upper = meta['upper']
        value = prob[var_name]
        if not (lower <= value <= upper):
            print(f"Design variable '{var_name}' is out of bounds: {value} not in [{lower}, {upper}]")
            raise ValueError(f"Design variable '{var_name}' is out of bounds: {value} not in [{lower}, {upper}]")


def optimize_magnetics_design(prob_in=None, output_dir=None, cleanup_flag=True, opt_flag=False, restart_flag=True, femm_flag=False, obj_str="levelized cost of energy", ratingkW=286):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(obj_str, obj_str.lower())
    ratingkW = int(ratingkW)
    target_torque = 1e6 * ratingkW/(np.pi*rated_speed[ratingkW]/30.0)/target_eff        # change or remove, this is based on wind turbines

    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = PMSG_Inner_rotor_Opt(magnetics_by_fea=femm_flag)

    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options["maxiter"] = 200
    prob.driver.options["tol"] = 1e-8
    #prob.driver = om.DifferentialEvolutionDriver()
    #prob.driver.options["max_gen"] = 15
    #prob.driver.options["pop_size"] = 30
    #prob.driver.options["penalty_exponent"] = 3

    #recorder = om.SqliteRecorder(os.path.join(output_dir, fsql))
    #prob.driver.add_recorder(recorder)
    #prob.add_recorder(recorder)
    #prob.driver.recording_options["excludes"] = ["*_df"]
    #prob.driver.recording_options["record_constraints"] = True
    #prob.driver.recording_options["record_desvars"] = True
    #prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("r_g", lower=0.5, upper=2.0)      # airgap radius
    prob.model.add_design_var("l_s", lower=0.5, upper=2.5)      # core length
    prob.model.add_design_var("h_s", lower=0.025, upper=0.1)      #, ref=0.01)  # yoke height
    prob.model.add_design_var("g", lower=0.006, upper=0.009)     #, ref=0.01)   # airgap length
    prob.model.add_design_var("h_yr", lower=0.01, upper=0.1)     #, ref=0.01)   # rotor yoke height
    prob.model.add_design_var("h_ys", lower=0.01, upper=0.1)     #, ref=0.01)   # stator yoke height
    prob.model.add_design_var("p", lower=4, upper=10)           # number of pole pairs
    prob.model.add_design_var("N_c", lower=2, upper=12)         # turns per coil
    prob.model.add_design_var("h_m", lower=0.005, upper=0.075)   #, ref=0.01)   # magnet height
    prob.model.add_design_var("I_s", lower=500, upper=6000)      #, ref=1e3)    # Stator current
    prob.model.add_design_var("ratio", lower=0.7, upper=0.85)   # ratio of magnet width to pole pitch
    prob.model.add_design_var("gear_ratio", lower=0.0)          # effective gear ratio


    #prob.model.add_constraint("E_p", upper=1.2 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", lower=0.9, upper=1.1)
    prob.model.add_constraint("torque_ratio", lower=1.05, upper=1.15)
    #prob.model.add_constraint("T_e", upper=1.2*target_torque, ref=1e5)
    prob.model.add_constraint("gen_eff", lower=0.96)
    prob.model.add_constraint("AEP", lower=0)
    #prob.model.add_constraint("LCOE", lower=0)

    if obj_str.lower() == 'cost':
        prob.model.add_objective("cost_total", ref=1e5)
    elif obj_str.lower() == 'mass':
        prob.model.add_objective("mass_total", ref=1e5)
    elif obj_str.lower() in ['eff','efficiency']:
        prob.model.add_objective("gen_eff", scaler=-1.0)
    elif obj_str.lower() == 'levelized cost of energy':
        prob.model.add_objective("LCOE", ref=1.0)
    else:
        print('Objective?', obj_str)

    prob.model.approx_totals(method="fd")

    prob.setup()
    print(prob.model.get_design_vars())
    print('************************')
    print('Design objectives:')
    print('Generator rating in MW: ', ratingkW)
    # print('Rated speed in rpm: ',rated_speed[ratingkW])   
    print('Target torque in MNm: ', target_torque*1e-6)
    print('Objective function: ', obj_str)
    print('************************')

    if prob_in is None:
        # Specific costs
        prob["resisitivty_Cu"] = 1.724e-8  # 1.8e-8 * 1.4  # Copper resisitivty
        prob["rho_Copper"] = 8900.0  # Kg/m3 copper density
        prob["rho_Fe"] = 7700.0  # Steel density
        prob["rho_Fes"] = 7850
        prob["rho_PM"] = 8442.37093661195  # magnet density

        # ## Initial design variables for a PMSG designed for a 15MW turbine
        prob["B_r"] = 1.279
        prob["E"] = 2.0e11
        prob["E_p_target"] = 3300.0
        prob["H_c"] = 1.0 #units="A/m", desc="coercivity"
        prob["I_s"] = 1945.9858772
        prob["J_s"] = 3.0
        prob["N_c"] = 3.0
        prob["P_Fe0e"] = 1.0
        prob["P_Fe0h"] = 4.0
        prob["R_no"] = 0.35
        prob["R_sh"] = 0.3
        prob["b_r"] = 0.200
        prob["b_s_tau_s"] = 0.45
        prob["b_st"] = 0.4

        prob["cost_adder"] = 0.0  # 700k, could maybe increase a bit at 25MW
        prob["mass_adder"] = 0.0   # 77t, could maybe increase a bit at 25MW

        prob["d_r"] = 0.500
        prob["d_s"] = 0.6
        prob["g"] = 0.008
        prob["g1"] = 9.806
        prob["h_m"] = 0.01
        prob["h_s"] = 0.05
        prob["h_s1"] = 0.010
        prob["h_s2"] = 0.010
        prob["h_sr"] = 0.02
        prob["h_ss"] = 0.02
        prob["h_yr"] = 0.05
        prob["h_ys"] = 0.05
        prob["k_fes"] = 0.95
        prob["k_sfil"] = 0.65
        prob["l_s"] = 0.6
        prob["m"] = 3  # phases
        prob["mu_0"] = np.pi * 4 * 1e-7
        prob["mu_r"] = 1.06
        prob["n_r"] = 6
        prob["n_s"] = 5
        prob["p"] = 6.0
        prob["phi"] = 90
        prob["q1"] = 1  # slots per pole per phase
        prob["r_g"] = 2.0
        prob["ratio"]= 0.7
        prob["gear_ratio"] = 0.5

        prob["t_s"] = 0.02
        prob["t_wr"] = 0.02
        prob["t_ws"] = 0.04
        prob["u_allow_pcent"] = 10
        prob["y_allow_pcent"] = 20
        prob["z_allow_deg"] = 0.5
        prob["gamma"] = 1.5

        if restart_flag:
            prob = load_data(os.path.join(output_dir, output_root), prob)

        # Have to set these last in case we initiatlized from a different rating
        prob["P_rated"] = ratingkW * 1e6
        prob["T_rated"] = target_torque
        prob["N_nom"] = rated_speed[ratingkW]
        prob["N_rated"] = rated_speed[ratingkW]
        prob["N_c"] = 3.0

        #Specific costs
        prob["C_Cu"]           = 7.3
        prob["C_Fe"]           = 1.56
        prob["C_Fes"]          = 4.44
        prob["C_PM"]           = 66.72
        prob["C_NbTi"]         = 45.43

    else:
        prob = copy_data(prob_in, prob)

    validate_inputs(prob)

    prob.model.approx_totals(method="fd")

    if opt_flag:
        prob.run_driver()
    else:
        prob.run_model()

    # Clean run directory after the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    #prob.model.list_outputs(val=True, hierarchical=True)
    print("Final solution:")
    print("E_p_ratio", prob["E_p_ratio"])
    print("gen_eff", prob["gen_eff"])
    print("N_c", prob["N_c"])
    print("l_s", prob["l_s"])
    print("Torque_ratio", prob["torque_ratio"])

    return prob

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    prob_struct = om.Problem()
    prob_struct.model = PMSG_Inner_Rotor_Structural()

    prob_struct.driver = om.ScipyOptimizeDriver()
    prob_struct.driver.options["optimizer"] = "SLSQP"
    prob_struct.driver.options["maxiter"] = 100

    #recorder = om.SqliteRecorder("log.sql")
    #prob_struct.driver.add_recorder(recorder)
    #prob_struct.add_recorder(recorder)
    #prob_struct.driver.recording_options["excludes"] = ["*_df"]
    #prob_struct.driver.recording_options["record_constraints"] = True
    #prob_struct.driver.recording_options["record_desvars"] = True
    #prob_struct.driver.recording_options["record_objectives"] = True

    prob_struct.model.add_design_var("h_sr", lower=0.045, upper=0.25)       # Rotor rim thickness
    prob_struct.model.add_design_var("h_ss", lower=0.045, upper=0.25)       # Stator rim thickness
    prob_struct.model.add_design_var("n_r", lower=5, upper=15)              # Rotor arms
    prob_struct.model.add_design_var("b_r", lower=0.1, upper=1.5)           # Rotor arm circumferential dimension
    prob_struct.model.add_design_var("d_r", lower=0.1, upper=1.5)           # Rotor arm depth
    prob_struct.model.add_design_var("t_wr", lower=0.001, upper=0.2)        # Rotor arm thickness
    prob_struct.model.add_design_var("n_s", lower=5, upper=15)              # Stator arms
    prob_struct.model.add_design_var("b_st", lower=0.1, upper=1.5)          # Stator arm circumferential dimension
    prob_struct.model.add_design_var("d_s", lower=0.1, upper=1.5)           # Stator arm depth
    prob_struct.model.add_design_var("t_ws", lower=0.001, upper=0.2)        # Stator arm thickness
    prob_struct.model.add_objective("mass_structural", ref=1e3)             

    prob_struct.model.add_constraint("con_bar", upper=1.0)
    prob_struct.model.add_constraint("con_uar", upper=1.0)
    prob_struct.model.add_constraint("con_yar", upper=1.0)
    prob_struct.model.add_constraint("con_zar", upper=1.0)

    prob_struct.model.add_constraint("con_bas", upper=1.0)
    prob_struct.model.add_constraint("con_uas", upper=1.0)
    prob_struct.model.add_constraint("con_yas", upper=1.0)
    prob_struct.model.add_constraint("con_zas", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()

    prob_struct = copy_data(prob_in, prob_struct)

    prob_struct.model.approx_totals(method="fd")

    if opt_flag:
        prob_struct.run_driver()
    else:
        prob_struct.run_model()

    return prob_struct

def write_all_data(prob, output_dir=None):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    save_data(os.path.join(output_dir, output_root), prob)
    print(prob.get_val("P_rated", units="MW"))
    ratingkW = float(prob.get_val("P_rated", units="MW")[0])
    raw_data = [
        ["Rating",                        "P_rated",                ratingkW, "MW", ""],
        ["specific current loading"                           , "A_1"               , float(prob.get_val("A_1", units="A/m")[0]), "A/m"],
        ["Peak air gap flux density"                          , "B_g"               , float(prob.get_val("B_g", units="T")[0]), "T"],
        ["remnant flux density"                               , "B_r"               , float(prob.get_val("B_r", units="T")[0]), "T"],
        ["Peak Rotor yoke flux density"                       , "B_rymax"           , float(prob.get_val("B_rymax", units="T")[0]), "T"],
        ["Peak Stator Yoke flux density"                      , "B_symax"           , float(prob.get_val("B_symax", units="T")[0]), "T"],
        ["Peak Teeth flux density"                            , "B_tmax"            , float(prob.get_val("B_tmax", units="T")[0]), "T"],
        ["no of slots per pole per phase"                     , "q1"                , float(prob.get_val("q1")[0]), "-"],
        ["Air-gap radius"                                     , "r_g"               , float(prob.get_val("r_g", units="m")[0]), "m", "(0.75-2.0)"],
        ["ratio of magnet width to pole pitch"                , "ratio"             , float(prob.get_val("ratio")[0]), "-", "(0.7-0.85)"],
        ["Copper resistivity"                                 , "resisitivty_Cu"    , float(prob.get_val("resisitivty_Cu", units="ohm*m")[0]), "ohm*m"],
        ["airgap length"                                      , "g"                 , float(prob.get_val("g", units="m")[0]), "m", "(0.06-0.09)"],
        ["magnet height"                                      , "h_m"               , float(prob.get_val("h_m", units="m")[0]), "m", "(0.005-0.075)"],
        ["Yoke height"                                        , "h_s"               , float(prob.get_val("h_s", units="m")[0]), "m", "(0.05-0.1)"],
        ["Slot Opening height"                                , "h_s1"              , float(prob.get_val("h_s1")[0]), "-"],
        ["Wedge Opening height"                               , "h_s2"              , float(prob.get_val("h_s2")[0]), "-"],
        ["Rotor yoke height"                                  , "h_yr"              , float(prob.get_val("h_yr", units="m")[0]), "m", "(0.01-0.1)"],
        ["Stator yoke height"                                 , "h_ys"              , float(prob.get_val("h_ys", units="m")[0]), "m", "(0.01-0.1)"],
        ["Stator outer diameter"                              , "D_out"             , float(prob.get_val("D_out", units="m")[0]), "m"],
        ["Stator current"                                     , "I_s"               , float(prob.get_val("I_s", units="A")[0]), "A", "(500-6000)"],
        ["Current density"                                    , "J_s"               , float(prob.get_val("J_s", units="A/mm**2")[0]), "A/mm**2"],
        ["core length"                                        , "l_s"               , float(prob.get_val("l_s", units="m")[0]), "m", "(0.5-2.5)"],
        ["no of phases"                                       , "m"                 , float(prob.get_val("m")[0]), "-"],
        ["No of pole pairs"                                   , "p"                 , float(prob.get_val("p")[0]), "-", "(4-10)"],
        ["Aspect ratio"                                       , "K_rad"             , float(prob.get_val("K_rad")[0]), "-"],
        ["Stator core length"                                 , "L_t"               , float(prob.get_val("L_t", units="m")[0]), "m"],
        ["turns per coil"                                     , "N_c"               , float(prob.get_val("N_c")[0]), "-", "(2-12)"],
        ["rated speed"                                        , "N_nom"             , float(prob.get_val("N_nom", units="rpm")[0]), "rpm"],
        ["turns per phase"                                    , "N_s"               , float(prob.get_val("N_s")[0]), "-"],
        ["magnet width"                                       , "b_m"               , float(prob.get_val("b_m", units="m")[0]), "m"],
        ["slot width"                                         , "b_s"               , float(prob.get_val("b_s", units="m")[0]), "m"],
        ["ratio of Slot width to slot pitch"                  , "b_s_tau_s"         , float(prob.get_val("b_s_tau_s")[0]), "-"],
        ["relative permeability of magnet"                    , "mu_r"              , float(prob.get_val("mu_r")[0]), "-"],
        ["Pole pitch"                                         , "tau_p"             , float(prob.get_val("tau_p", units="m")[0]), "m"],
        ["Stator slot pitch"                                  , "tau_s"             , float(prob.get_val("tau_s", units="m")[0]), "m"],
        ["Stator resistance"                                  , "R_s"               , float(prob.get_val("R_s", units="ohm")[0]), "ohm"],
        ["Stator slots"                                       , "S"                 , float(prob.get_val("S")[0]), "-"],
        #["Slot aspect ratio"                                  , "Slot_aspect_ratio" , float(prob.get_val("Slot_aspect_ratio")[0]), "-"],
        ["Stator phase voltage"                               , "E_p"               , float(prob.get_val("E_p", units="V")[0]), "V"],
        ["Voltage constraint"                                 , "E_p_ratio"         , float(prob.get_val("E_p_ratio")[0]), "", "0.8 < x < 1.2"],
        ["Target voltage"                                     , "E_p_target"        , float(prob.get_val("E_p_target", units="V")[0]), "V"],
        ["Normal stress"                                      , "Sigma_normal"      , float(prob.get_val("Sigma_normal", units="kN/m**2")[0]), "kN/m**2"],
        ["Shear stress"                                       , "Sigma_shear"       , float(prob.get_val("Sigma_shear", units="kN/m**2")[0]), "kN/m**2"],
        ["Electromagnetic torque"                             , "T_e"               , float(prob.get_val("T_e", units="kN*m")[0]), "kN*m"],
        ["Rated torque"                                       , "T_rated"           , float(prob.get_val("T_rated", units="kN*m")[0]), "kN*m"],
        ["torque constraint"                                  , "torque_ratio"      , float(prob.get_val("torque_ratio")[0]), "-", "1.0 < x < 1.2"],
        ["Copper losses"                                      , "P_Cu"              , float(prob.get_val("P_Cu", units="kW")[0]), "kW"],
        ["specific eddy losses W/kg @ 1.5 T"                  , "P_Fe0e"            , float(prob.get_val("P_Fe0e")[0]), "-"],
        ["specific hysteresis losses W/kg @ 1.5 T"            , "P_Fe0h"            , float(prob.get_val("P_Fe0h")[0]), "-"],
        ["Magnet losses"                                      , "P_Ftm"             , float(prob.get_val("P_Ftm", units="kW")[0]), "kW"],
        ["Total loss"                                         , "Losses"            , float(prob.get_val("Losses", units="kW")[0]), "kW"],
        ["Generator efficiency"                               , "gen_eff"           , float(prob.get_val("gen_eff")[0]), "", "0.95 < x"],
        ["Generator output frequency"                         , "f"                 , float(prob.get_val("f", units="Hz")[0]), "Hz"],
        ["Iron stacking factor"                               , "k_fes"             , float(prob.get_val("k_fes")[0]), "-"],
        ["slot fill factor"                                   , "k_sfil"            , float(prob.get_val("k_sfil")[0]), "-"],
        ["winding factor"                                     , "k_wd"              , float(prob.get_val("k_wd")[0]), "-"],
        ["Bedplate nose outer radius"                         , "R_no"              , float(prob.get_val("R_no", units="m")[0]), "m"],
        ["Main shaft outer radius"                            , "R_sh"              , float(prob.get_val("R_sh", units="m")[0]), "m"],
        ["Main shaft tilt angle"                              , "phi"               , float(prob.get_val("phi", units="deg")[0]), "deg"],
        ["rotor arm circumferential dimension"                , "b_r"               , float(prob.get_val("b_r", units="m")[0]), "m", "(0.1-1.5)"],
        ["stator arm circumferential dimension"               , "b_st"              , float(prob.get_val("b_st", units="m")[0]), "m", "(0.1-1.5)"],
        ["tooth width"                                        , "b_t"               , float(prob.get_val("b_t", units="m")[0]), "m"],
        ["rotor arm depth"                                    , "d_r"               , float(prob.get_val("d_r", units="m")[0]), "m", "(0.1-1.5)"],
        ["stator arm depth"                                   , "d_s"               , float(prob.get_val("d_s", units="m")[0]), "m", "(0.1-1.5)"],
        ["Rotor rim thickness"                                , "h_sr"              , float(prob.get_val("h_sr", units="m")[0]), "m", "(0.045-0.25)"],
        ["Stator rim thickness"                               , "h_ss"              , float(prob.get_val("h_ss", units="m")[0]), "m", "(0.045-0.25)"],
        ["Rotor arms"                                         , "n_r"               , float(prob.get_val("n_r")[0]), "-", "(5-15)"],
        ["Stator arms"                                        , "n_s"               , float(prob.get_val("n_s")[0]), "-", "(5-15)"],
        ["rotor arm thickness"                                , "t_wr"              , float(prob.get_val("t_wr", units="m")[0]), "m", "(0.001-0.2)"],
        ["stator arm thickness"                               , "t_ws"              , float(prob.get_val("t_ws", units="m")[0]), "m", "(0.001-0.2)"],
        ["Stator disc thickness"                              , "t_s"               , float(prob.get_val("t_s", units="m")[0]), "m"],
        ["Partial safety factor"                              , "gamma"             , float(prob.get_val("gamma")[0]), "", ""],
        ["Rotor radial deflection"                            , "u_ar"              , float(prob.get_val("u_ar", units="m")[0]), "m"],
        ["Stator radial deflection"                           , "u_as"              , float(prob.get_val("u_as", units="m")[0]), "m"],
        ["Allowable radial deflection percent"                , "u_allow_pcent"     , float(prob.get_val("u_allow_pcent")[0]), "-"],
        ["Rotor axial deflection"                             , "y_ar"              , float(prob.get_val("y_ar", units="m")[0]), "m"],
        ["Stator axial deflection"                            , "y_as"              , float(prob.get_val("y_as", units="m")[0]), "m"],
        ["Allowable axial deflection percent"                 , "y_allow_pcent"     , float(prob.get_val("y_allow_pcent")[0]), "-"],
        ["Rotor circumferential deflection"                   , "z_ar"              , float(prob.get_val("z_ar", units="m")[0]), "m"],
        ["Stator circumferential deflection"                  , "z_as"              , float(prob.get_val("z_as", units="m")[0]), "m"],
        ["Allowable torsional twist"                          , "z_allow_deg"       , float(prob.get_val("z_allow_deg", units="deg")[0]), "deg"],
        ["Circumferential arm space constraint"               , "con_bar"           , float(prob.get_val("con_bar")[0]), "-", "< 1.0"],
        ["Circumferential arm space constraint"               , "con_bas"           , float(prob.get_val("con_bas")[0]), "-", "< 1.0"],
        ["Radial deflection constraint-rotor"                 , "con_uar"           , float(prob.get_val("con_uar")[0]), "-", "< 1.0"],
        ["Radial deflection constraint-rotor"                 , "con_uas"           , float(prob.get_val("con_uas")[0]), "-", "< 1.0"],
        ["Axial deflection constraint-rotor"                  , "con_yar"           , float(prob.get_val("con_yar")[0]), "-", "< 1.0"],
        ["Axial deflection constraint-rotor"                  , "con_yas"           , float(prob.get_val("con_yas")[0]), "-", "< 1.0"],
        ["Torsional deflection constraint-rotor"              , "con_zar"           , float(prob.get_val("con_zar")[0]), "-", "< 1.0"],
        ["Torsional deflection constraint-rotor"              , "con_zas"           , float(prob.get_val("con_zas")[0]), "-", "< 1.0"],
        ["Copper density"                                     , "rho_Copper"        , float(prob.get_val("rho_Copper", units="kg/m**3")[0]), "kg/m**3"],
        ["Electrical steel density"                           , "rho_Fe"            , float(prob.get_val("rho_Fe", units="kg/m**3")[0]), "kg/m**3"],
        ["Structural Steel densit"                            , "rho_Fes"           , float(prob.get_val("rho_Fes", units="kg/m**3")[0]), "kg/m**3"],
        ["magnet mass density"                                , "rho_PM"            , float(prob.get_val("rho_PM", units="kg/m**3")[0]), "kg/m**3"],
        ["Rotor yoke mass"                                    , "M_Fery"            , float(prob.get_val("M_Fery", units="t")[0]), "t"],
        ["Stator teeth mass"                                  , "M_Fest"            , float(prob.get_val("M_Fest", units="t")[0]), "t"],
        ["Stator yoke mass"                                   , "M_Fesy"            , float(prob.get_val("M_Fesy", units="t")[0]), "t"],
        ["Iron mass"                                          , "mass_iron"              , float(prob.get_val("mass_iron", units="t")[0]), "t"],
        ["Iron mass"                                          , "mass_Fe"           , float(prob.get_val("mass_Fe", units="t")[0]), "t"],
        ["Magnet mass"                                        , "mass_PM"           , float(prob.get_val("mass_PM", units="t")[0]), "t"],
        ["Copper Mass"                                        , "mass_copper"            , float(prob.get_val("mass_copper", units="t")[0]), "t"],
        ["Active mass"                                        , "mass_active"       , float(prob.get_val("mass_active", units="t")[0]), "t"],
        ["Mass to add to total for unaccounted elements"      , "mass_adder"        , float(prob.get_val("mass_adder", units="t")[0]), "t"],
        ["Rotor structural mass"                              , "mass_structural_rotor"  , float(prob.get_val("mass_structural_rotor", units="t")[0]), "t"],
        ["Stator structural mass"                             , "mass_structural_stator" , float(prob.get_val("mass_structural_stator", units="t")[0]), "t"],
        ["Total structural mass"                              , "mass_structural"   , float(prob.get_val("mass_structural", units="t")[0]), "t"],
        ["Specific cost of copper"                            , "C_Cu"              , float(prob.get_val("C_Cu", units="USD/kg")[0]), "USD/kg"],
        ["Specific cost of magnetic steel/iron"               , "C_Fe"              , float(prob.get_val("C_Fe", units="USD/kg")[0]), "USD/kg"],
        ["Specific cost of structural steel"                  , "C_Fes"             , float(prob.get_val("C_Fes", units="USD/kg")[0]), "USD/kg"],
        ["Specific cost of Magnet"                            , "C_PM"              , float(prob.get_val("C_PM", units="USD/kg")[0]), "USD/kg"],
        ["Cost to add to total for unaccounted elements"      , "cost_adder"        , 1e-3*float(prob.get_val("cost_adder", units="USD")[0]), "k$"],
        ["Total mass"                                         , "mass_total"        , float(prob.get_val("mass_total", units="t")[0]), "t"],
        ["Total cost"                                         , "cost_total"        , 1e-3*float(prob.get_val("cost_total", units="USD")[0]), "k$"],
        ["Levelized Cost Of Energy"                           , "LCOE"              , float(prob.get_val("LCOE", units="USD/kWh")[0]), "USD/kWh"]
    ]

    df = pd.DataFrame(raw_data, columns=["Parameters", "Symbol", "Values", "Units", "Limit"])
    df.to_excel(os.path.join(output_dir, f"Optimized_MS-PMSG_{ratingkW}_MW.xlsx"))

# def get_eff_curve(output_str, obj_str, ratingkW):
#     output_dir = os.path.join(mydir, output_str)

#     prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=False, obj_str=obj_str,
#                                      ratingkW=int(ratingkW), restart_flag=True, femm_flag=True, cleanup_flag=False)

#     rpm = np.unique( np.minimum(rated_speed[int(ratingkW)], gear_ratio * np.arange(2, 8.1, 0.5)) )
#     torque = np.zeros(rpm.shape)
#     shear = np.zeros(rpm.shape)
#     normal = np.zeros(rpm.shape)
#     losses = np.zeros(rpm.shape)
#     eff = np.zeros(rpm.shape)
#     for ir, r in enumerate(rpm):
#         prob["N_nom"] = r
#         prob.run_model()
        
#         torque[ir] = float(prob.get_val("T_e",units="MN*m"))
#         shear[ir] = float(prob.get_val("Sigma_shear",units="kN/m**2"))
#         normal[ir] = float(prob.get_val("Sigma_normal",units="kN/m**2"))
#         losses[ir] = float(prob.get_val("Losses",units="kW"))
#         eff[ir] = float(prob.get_val("gen_eff"))

#     np.savetxt(os.path.join(output_dir, f"eff_curve_{ratingkW}MW-MSPMSG.csv"),
#                np.c_[rpm, torque, shear, normal, losses, eff],
#                delimiter=',',
#                header='RPM,Torque [MNm],Shear stress [kN/m^2],Normal stress [kN/m^2],Losses [kW],Efficiency')
    
def run_all(output_str, opt_flag, obj_str, ratingkW):
    output_dir = os.path.join(mydir, output_str)

    # Optimize just magnetrics with GA and then structural with SLSQP
    prob=None
    #prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=True, obj_str=obj_str, ratingkW=int(ratingkW), restart_flag=True, femm_flag=False)
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str, ratingkW=int(ratingkW),
                                     prob_in=prob, restart_flag=True, femm_flag=True, cleanup_flag=False)
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=True)

    # Bring all data together
    for k in ["h_sr","h_ss","n_r","n_s","b_r","d_r","t_wr","t_ws","b_st","d_s"]:
        prob[k]  = prob_struct[k]
    prob.run_model()

    # Write to xlsx and csv files
    #prob.model.list_inputs(val=True, hierarchical=True, units=True, desc=True)
    #prob.model.list_outputs(val=True, hierarchical=True)
    write_all_data(prob, output_dir=output_dir)
    cleanup_femm_files(mydir, output_dir)

if __name__ == "__main__":
    opt_flag = True
    #run_all("outputs15-mass", opt_flag, "mass", 15)
    #run_all("outputs17-mass", opt_flag, "mass", 17)
    #run_all("outputs20-mass", opt_flag, "mass", 20)
    #run_all("outputs25-mass", opt_flag, "mass", 25)
    #run_all("outputs15-cost", opt_flag, "cost", 15)
    #run_all("outputs17-cost", opt_flag, "cost", 17)
    #run_all("outputs20-cost", opt_flag, "cost", 20)
    #run_all("outputs22-cost", opt_flag, "cost", 22)
    run_all("outputs25-LCOE", opt_flag, "levelized cost of energy", 286)
    # for k in ratings_known:
    #     for obj in ["LCOE"]:#, "mass"]:
    #         for m in range(2):
    #             run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
    #for k in ratings_known:
    #    get_eff_curve(f"outputs{k}-cost", "cost", k)
