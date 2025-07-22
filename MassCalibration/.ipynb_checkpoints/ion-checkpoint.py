import re
from .constants import amu, me
class Ion:
    def __init__(self, element, Z_ion, A_ion, Q_ion):
        self.element = re.sub(r'\d+', '', element)  # Remove digits from the element
        self.Z_ion = Z_ion  # Atomic number
        self.A_ion = A_ion  # Mass number
        self.Q_ion = Q_ion  # Charge state
        self.revolution_time = None  # Revolution time in ns
        self.sigmaRevT = None  # widht of peakin ps
        self.revolution_time_error = None  # Error in revolution time in ps
        self.mass_excess = 0  # Mass excess in MeV
        self.mass_excess_error = 0  # Mass excess error in MeV
        self.binding_energy = 0  # Binding energy in MeV
        self.mass = None  # Mass in MeV
        self.mass_Q = None  # Mass/amu/Q
        self.mass_Q_error = None  # Mass_Q error
        self.is_reference_ion = False  # Whether the ion is a reference ion, default to False
        self.exp_mass_excess = None  # Experimental mass excess in MeV
        self.exp_mass_excess_error = None  # Experimental mass excess error in MeV
        self.exp_mass_excess_error_fitav = None  # Experimental mass excess error in MeV
        self.exp_mass_excess_error_freq = None  # Experimental mass excess error in MeV
        self.exp_mass_Q = None  # Experimental mass/amu/Q
        self.exp_mass_Q_error = None  # Experimental mass/amu/Q error
        self.initial_params = None 
        self.initial_covariance_matrix = None
        self.params = None  # Add beta to store fit parameters for each ion
        self.covariance_matrix = None
        self.chi2 = None
        
    def set_revolution_time(self, revolution_time, sigmaRevT, revolution_time_error):
        self.revolution_time = revolution_time
        self.sigmaRevT = sigmaRevT
        self.revolution_time_error = revolution_time_error

    def set_mass_excess(self, mass_excess, mass_excess_error):
        self.mass_excess = mass_excess
        self.mass_excess_error = mass_excess_error

    def set_binding_energy(self, binding_energy):
        self.binding_energy = binding_energy

    def set_exp_mass_excess(self, exp_mass_excess, exp_mass_excess_error):
        self.exp_mass_excess = exp_mass_excess
        self.exp_mass_excess_error = exp_mass_excess_error

    def calculate_mass_properties(self):
        if self.A_ion is not None:
            self.mass = self.A_ion * amu + self.mass_excess - self.Q_ion * me + self.binding_energy
            self.mass_Q = self.mass / amu / self.Q_ion
            if self.mass != 0:
                self.mass_Q_error = (self.mass_excess_error / self.mass) * self.mass_Q
            else:
                self.mass_Q_error = None

            if self.exp_mass_excess is not None:
                self.exp_mass_Q = (self.A_ion * amu + self.exp_mass_excess - self.Q_ion * me + self.binding_energy) / amu / self.Q_ion
                self.exp_mass_Q_error = (self.exp_mass_excess_error / (self.A_ion * amu + self.exp_mass_excess - self.Q_ion * me + self.binding_energy)) * self.exp_mass_Q

    def __repr__(self):
        return (f"Ion({self.element}, Z={self.Z_ion}, A={self.A_ion}, Q={self.Q_ion}, "
                f"RevolutionTime={self.revolution_time} ns, "
                f"RevolutionTimeError={self.revolution_time_error} ps, "
                f"MassExcess={self.mass_excess} MeV, "
                f"MassExcessError={self.mass_excess_error} MeV, "
                f"BindingEnergy={self.binding_energy} MeV, "
                f"Mass={self.mass} MeV, "
                f"Mass_Q={self.mass_Q}, "
                f"Mass_Q_Error={self.mass_Q_error}, "
                f"IsReferenceIon={self.is_reference_ion}, "
                f"ExpMassExcess={self.exp_mass_excess} MeV, "
                f"ExpMassExcessError={self.exp_mass_excess_error} MeV, "
                f"ExpMass_Q={self.exp_mass_Q}, "
                f"ExpMass_Q_Error={self.exp_mass_Q_error})")
