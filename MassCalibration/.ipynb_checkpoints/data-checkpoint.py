import mpmath as mp
import numpy as np
import re
from MassCalibration.ion import Ion
from prettytable import PrettyTable
class Data:
    def __init__(self, ame_file="mass16.rd", elbien_file="ElBiEn_2007.dat", revtime_file="revlutiontime_ECIMS.txt",p=2):
        self.ame_file = ame_file
        self.elbien_file = elbien_file
        self.revtime_file = revtime_file
        self.p = p
        self.ions = []
        self.chi2_mq = None
        self.sigma_syst_mq = None
        self.chi2_me = None
        self.sigma_syst_me = None

        self.read_revolution_time(self.revtime_file)
        self.read_ame(self.ame_file)
        self.read_elbien_file(self.elbien_file)
        self.calculate_all_ion_masses()
        self.T, self.MoQ, self.TError, self.MoQError = self.extract_data()

    def read_revolution_time(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return
        print(f"Reading revolution time from file '{file_path}'...")

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                tokens = line.split()
                element_with_mass = tokens[0]
                A = int(re.match(r'\d+', element_with_mass).group())  # Extract leading digits as mass number
                element = re.sub(r'\d+', '', element_with_mass)  # Remove digits from the element
                Z = int(tokens[1])
                Q = int(tokens[2])
                is_reference_ion = tokens[3].strip() == 'Y'  # Check if the 4th column is 'Y'
                revolution_time = float(tokens[4])
                sigmaRevT = float(tokens[6])
                revolution_time_error = float(tokens[7])
                ion = Ion(element, Z, A, Q)
                ion.set_revolution_time(revolution_time, sigmaRevT, revolution_time_error)
                ion.is_reference_ion = is_reference_ion  # Set the is_reference_ion attribute

                self.ions.append(ion)
                #print(f"Debug: IsReferenceIon for {ion.element}-{ion.A_ion}: {ion.is_reference_ion}")  # Debug output
                
    def read_ame(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return
        print(f"Reading mass data from file '{file_path}'...")

        for line in lines:
            line = line.strip()
            if line:
                tokens = line.split()
                if len(tokens) == 6:
                    element_with_mass = tokens[0]
                    A = int(re.match(r'\d+', element_with_mass).group())  # Extract leading digits as mass number
                    element = re.sub(r'\d+', '', element_with_mass)  # Remove digits from the element
                    Z = int(tokens[1])
                    ME = float(tokens[3]) / 1e3  # MeV
                    MEError = float(tokens[4]) / 1e3  # MeV
                    for ion in self.ions:
                        if ion.element == element and ion.Z_ion == Z and ion.A_ion == A:
                            ion.set_mass_excess(ME, MEError)
                            #print(f"Updated ion after read_ame: {ion}")  # Output updated ion information
                            break

    def read_elbien_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return
        print(f"Reading electron binding energy data from file '{file_path}'...")

        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('#'):
                if header_line is None:
                    header_line = line
                continue
            
            if not line.strip():
                continue
                
            tokens = line.split()
            if header_line:
                headers = header_line.strip().split()[1:]
                Z = int(tokens[0])
                for Q in range(1, Z + 1):
                    binding_energy = int(tokens[Q + 1]) / 1e6  # MeV
                    for ion in self.ions:
                        if ion.Z_ion == Z and ion.Q_ion == Q:
                            ion.set_binding_energy(binding_energy)
                            #print(f"Updated ion after read_elbien_file: {ion}")  # Output updated ion information
                            break

    def calculate_all_ion_masses(self):
        for ion in self.ions:
            ion.calculate_mass_properties()

    def extract_data(self):
        T = [ion.revolution_time for ion in self.ions if ion.revolution_time is not None]
        MoQ = [ion.mass_Q for ion in self.ions if ion.mass_Q is not None]
        TError = [ion.revolution_time_error for ion in self.ions if ion.revolution_time_error is not None]
        MoQError = [ion.mass_Q_error for ion in self.ions if ion.mass_Q_error is not None]
        return T, MoQ, TError, MoQError

    def get_ions(self):
        return self.ions

    def calculate_chi2_and_systematic_error(self, p):
        reference_ions = [ion for ion in self.ions if ion.is_reference_ion]
        self.p = p
        self.chi2_me = 0
        self.sigma_syst_me = 0
        
        # Calculate initial chi2_me
        for ion in reference_ions:
            if ion.exp_mass_excess is None or ion.mass_excess is None:
                continue

            delta_me = ion.exp_mass_excess - ion.mass_excess
            numerator_me = delta_me ** 2
            denominator_me = ion.exp_mass_excess_error ** 2 + ion.mass_excess_error ** 2
            self.chi2_me += numerator_me / denominator_me
            #print("delta_me ",delta_me," denominator_me ",denominator_me," ion.exp_mass_excess_error = ",ion.exp_mass_excess_error," ion.mass_excess_error = ",ion.mass_excess_error, " self.chi2_me ",self.chi2_me)
        self.chi2_me = mp.sqrt(self.chi2_me / (len(reference_ions) - p))
        #print("self.chi2_me",self.chi2_me)
        # Iteratively adjust sigma_syst_me
        for i in range(100):
            chi2_me_iter = 0
            for ion in reference_ions:
                if ion.exp_mass_excess is None or ion.mass_excess is None:
                    continue

                delta_me = ion.exp_mass_excess - ion.mass_excess
                numerator_me = delta_me ** 2
                denominator_me = ion.exp_mass_excess_error ** 2 + ion.mass_excess_error ** 2 + self.sigma_syst_me ** 2

                chi2_me_iter += numerator_me / denominator_me
            #print(chi2_me_iter,"len(reference_ions) - p) ",len(reference_ions) - p)
            chi2_me_iter = mp.sqrt(chi2_me_iter / (len(reference_ions) - p))
            #print(chi2_me_iter,"len(reference_ions) - p) ",len(reference_ions) - p)
            if float(chi2_me_iter) - 1.0 < 0:
                break

            self.sigma_syst_me = float(self.sigma_syst_me) + 0.001

        return float(self.chi2_me), float(self.sigma_syst_me)
    
    def print_final_experimental_me(self):
        # Print final experimental ME and error for all ions
        print("\nFinal experimental ME and error for ATOM unit (keV)\n")
        print(f"Fit order: p, (m/q = 1 + a1*T + a2*T^2 + ... a_p*T^p) {self.p}")
        print(f"REF_NUC NO.            {len([ion for ion in self.get_ions() if ion.is_reference_ion])}")

        table = PrettyTable()
        table.field_names = ["Nuc", "Ref_nuc", "TOF/ps", "ME(EXP-AME)/keV", "ME_EXP/MeV", "ER.(EXP)/keV", "Fit ER.(EXP)", "Freq. ER.(EXP)", "ER.(AME)"]

        # Set alignment for numerical columns to be right-aligned
        table.align["TOF/ps"] = "r"
        table.align["ME(EXP-AME)/keV"] = "r"
        table.align["ME_EXP/MeV"] = "r"
        table.align["ER.(EXP)/keV"] = "r"
        table.align["Fit ER.(EXP)"] = "r"
        table.align["Freq. ER.(EXP)"] = "r"
        table.align["ER.(AME)"] = "r"

        for ion in self.get_ions():
            if ion.exp_mass_excess is not None and ion.mass_excess is not None:
                exp_me_diff = float((ion.exp_mass_excess - ion.mass_excess) * 1000)  # ME(EXP-AME)
                revolution_time = float(ion.revolution_time)
                exp_mass_excess = float(ion.exp_mass_excess)
                exp_mass_excess_error = float(ion.exp_mass_excess_error * 1000)
                exp_mass_excess_error_fitav = float(ion.exp_mass_excess_error_fitav * 1000)
                exp_mass_excess_error_freq = float(ion.exp_mass_excess_error_freq * 1000)
                mass_excess_error = float(ion.mass_excess_error * 1000)

                ion_name = f"{ion.A_ion}{ion.element}{ion.Q_ion}+"
                ref_nuc = 'Y' if ion.is_reference_ion else 'N'

                table.add_row([
                    ion_name, ref_nuc, f"{revolution_time:.5f}", 
                    f"{exp_me_diff:.3f}", f"{exp_mass_excess:.3f}", 
                    f"{exp_mass_excess_error:.3f}", f"{exp_mass_excess_error_fitav:.3f}", 
                    f"{exp_mass_excess_error_freq:.3f}", f"{mass_excess_error:.3f}"
                ])

        print(table)

        if self.chi2_me is not None and self.sigma_syst_me is not None:
            print(f"Chi2: {float(self.chi2_me):.2f}, Systematic error: {float(self.sigma_syst_me) * 1000:.2f} keV")

        



