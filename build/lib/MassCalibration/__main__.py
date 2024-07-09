import mpmath as mp
import numpy as np
import ROOT
from MassCalibration.ion import Ion
from MassCalibration.data import Data
from MassCalibration.model import Model
from MassCalibration.gui import GUI

def main():
    ame_file="mass16.rd"
    elbien_file="ElBiEn_2007.dat"
    #revtime_file="revlutiontime_ECIMS.txt"
    revtime_file="revlutiontime_72Ge.txt"
    p = 2
    iterationMax = 200
    OutputMatrixOrNot = True
    data = Data(ame_file,elbien_file,revtime_file,p)
    initial_params = [-0.6880425289251325, 2.9730599061989426e-06, 5.55756235663825e-12]
    model = Model(data, p,iterationMax,OutputMatrixOrNot,initial_params)
    model.calibration()  # Perform the fit before drawing the graph
    model.self_calibration()  # Perform the fit before drawing the graph
    data.calculate_chi2_and_systematic_error(p)  # Calculate chi2 and systematic error
    data.print_final_experimental_me()  # Print final experimental ME and error for all ions
    gui = GUI(data, model)
    gui.draw_individual_fits()
    gui.draw_me_vs_tof()  # Draw ME(EXP-AME) vs TOF    
if __name__ == "__main__":
    main()
