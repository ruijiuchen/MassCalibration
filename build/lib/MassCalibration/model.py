import mpmath as mp
import numpy as np
import ROOT
from ROOT import TMatrixTSym
from .constants import amu, me
import mpmath
from prettytable import PrettyTable

class Model:
    def __init__(self, data, p,iterationMax,OutputMatrixOrNot,initial_params):
        self.data = data
        self.p = p
        self.iterationMax = iterationMax
        self.OutputMatrixOrNot = OutputMatrixOrNot
        self.initial_params = initial_params
        self.initial_covariance_matrix = None
        self.initial_chi2 = None
        self.params = None
        self.covariance_matrix = None
        self.chi2 = None
        
    def get_initial_params_from_root(self, T, MoQ):
        # Create a TGraphErrors for ROOT fitting
        graph = ROOT.TGraphErrors(len(T))
        for i in range(len(T)):
            graph.SetPoint(i, T[i], MoQ[i])

        # Perform a polynomial fit of degree p
        fit = ROOT.TF1("fit", f"pol{self.p}", min(T), max(T))
        for i in range(self.p + 1):
            if i < len(self.initial_params):
                fit.SetParameter(i, self.initial_params[i])
            else:
                fit.SetParameter(i, 0)  # Set parameter to 0 if i exceeds the length of initial_params
        fitResult = graph.Fit(fit, "S Q")  # Quiet mode
        
        # Get the fit parameters
        initial_params = [fit.GetParameter(i) for i in range(self.p + 1)]
        
        # Extract the covariance matrix
        covariance_matrix_root = fitResult.GetCovarianceMatrix()
        
        covariance_matrix = []
        for i in range(covariance_matrix_root.GetNrows()):
            row = []
            for j in range(covariance_matrix_root.GetNcols()):
                row.append(covariance_matrix_root[i][j])
            covariance_matrix.append(row)
        
        return initial_params,covariance_matrix

    def LeastSquareFit(self, T, TError, MoQ, MoQError, p, iterationMax, initial_params, initial_covariance_matrix, OutputMatrixOrNot):
        print("##LeastSquareFit")
        mpmath.mp.dps = 500  # 500 decimal places precision
        A = initial_params[:]
        A_min = A[:]
        b2_inverse_min = initial_covariance_matrix
        chi2_min = 1e20

        iteration = 0
        N = len(T)
        for iteration in range(0, iterationMax):
            if OutputMatrixOrNot: 
                print("### iteration = ", iteration)
            FDB = mpmath.matrix(N, p)
            PDB = mpmath.matrix(N, N)
            delta_MoQ_fiDB = mpmath.matrix(N, 1)
            # step 1: F matrix.
            for i in range(0, N):
                f = T[i]
                for k in range(0, p):
                    FDB[i, k] = f ** k
            # step 2: delta_MoQ_fi
            for i in range(0, N):
                fi = T[i]
                delta_fi = TError[i]
                delta_MoQ_fi = 0
                for k in range(1, p):
                    delta_MoQ_fi = delta_MoQ_fi + k * A[k] * fi ** (k - 1) * delta_fi
                delta_MoQ_fiDB[i] = delta_MoQ_fi
            # step 3: P matrix
            for i in range(0, N):
                for j in range(0, N):
                    a = 0
                    if i == j:
                        a = 1.0 / (float(MoQError[i]) ** 2 + float(delta_MoQ_fiDB[i]) ** 2)
                    PDB[i, j] = a

            # step 4: chi2
            if iteration == 0:
                chi2 = 0
                for i in range(0, N):
                    fi = T[i]
                    delta_fi = TError[i]
                    y = MoQ[i]
                    ye = MoQError[i]
                    yfit = 0
                    yfit_error = delta_MoQ_fiDB[i]
                    for k in range(0, p):
                        yfit = yfit + A[k] * fi ** k
                    chi2 = chi2 + (y - yfit) ** 2 / (ye ** 2 + yfit_error ** 2)
                chi2_min = chi2
                A_min = A[:]   # don't change this line!
            if OutputMatrixOrNot:
                if iteration < 4:
                    print("1.A=")
                    for k in range(0, p):
                        if len(A) > 0:
                            print("%20.6e" % A[k])
                        else:
                            print("None.")
                    print("2.MoQDB +/- MoQDBError +/- delta_MoQ_fi | T +/- TError")
                    for i in range(0, N):
                        print("%10.5f" % MoQ[i], " +/- %10.5e" % MoQError[i], " +/- %10.5e" % delta_MoQ_fiDB[i], " | %10.5f" % T[i], " +/- %10.5f" % TError[i])

            # step 5: calculate parameters A
            F = FDB
            P = PDB
            MoQ_mpmath = mpmath.matrix([[v] for v in MoQ])  # Ensure MoQ_mpmath is a column vector
            F_T = F.T  # Calculate the transpose of F

            if OutputMatrixOrNot:
                if iteration < 4:
                    print("3.F=")
                    for i in range(0, N):
                        for k in range(0, p):
                            print("%10.3f" % F[i, k], end="\t")
                        print()
                    print("3.b F_T=")
                    for i in range(0, p):
                        for k in range(0, N):
                            print("%5.3e" % F_T[i, k], end="\t")
                        print()
                    print("4. P=")
                    for i in range(0, N):
                        for k in range(0, N):
                            if i == k:
                                print("%6.3f" % P[i, k], end="\t")
                            else:
                                print("%1d" % P[i, k], end="\t")
                        print()

            a1 = P * MoQ_mpmath
            a2 = F_T * a1
            b1 = P * F
            b2 = F_T * b1

            if OutputMatrixOrNot:
                if iteration < 4:
                    print("5.a1 = P*MoQ")
                    for i in range(0, N):
                        print("%10.5f" % a1[i])
                    print("6.a2 = F_T*P*MoQ)")
                    for i in range(0, p):
                        print("%10.5f" % a2[i])
                    print("7. b1=P*F")
                    for i in range(0, N):
                        for k in range(0, p):
                            print("%6.3f" % b1[i, k], end="\t")
                        print()
                    print("8. b2=F_T*P*F")
                    for i in range(0, p):
                        for k in range(0, p):
                            print("%30.3f" % b2[i, k], end="\t")
                        print()

            # Calculate the determinant of the matrix
            det_b2 = mpmath.det(b2)

            chi2 = 0        
            if det_b2 == 0:
                if OutputMatrixOrNot: print("b2 = FT*P*F is a singular matrix. Iteration stop.")
                break
            else:
                # Calculate the inverse of the matrix
                b2_inverse = mpmath.inverse(b2)
                A = b2_inverse * a2

                if iteration == 0: 
                    b2_inverse_min = b2_inverse.copy()
                # step 6
                for i in range(0, N):
                    fi = T[i]
                    delta_fi = TError[i]
                    y = MoQ[i]
                    ye = MoQError[i]
                    yfit = 0
                    yfit_error = delta_MoQ_fiDB[i]
                    for k in range(0, p):
                        yfit = yfit + A[k] * fi ** k
                    chi2 = chi2 + (y - yfit) ** 2 / (ye ** 2 + yfit_error ** 2)
                if chi2_min >= chi2:
                    chi2_min = chi2
                    A_min = A.copy()
                    b2_inverse_min = b2_inverse.copy()
                    iteration = iteration + 1
            # step 7
            #############################plot matrix#############################################
            if OutputMatrixOrNot:
                if iteration < 4:
                    print("9. b2_inverse=(F_T*P*F)^-1")
                    for i in range(0, p):
                        for k in range(0, p):
                            print("%30.3e" % b2_inverse[i, k], end="\t")
                        print()
                    print("10.A=")
                    for i in range(0, p):
                        print("%20.6e" % A[i])
                    print(" chi2    = %10.5e" % chi2)
            if OutputMatrixOrNot: 
                # Create a table
                table = PrettyTable()
                table.field_names = ["chi2", "chi2_min"]

                # Add chi2 and chi2_min to the table
                table.add_row(["%10.5e" % chi2, "%10.5e" % chi2_min])

                # Print the table for chi2 and chi2_min
                print(table)

                # Create a new table for A and A_min
                table_A = PrettyTable()
                table_A.field_names = ["A[i]", "A_min[i]"]

                # Add each pair of A[i] and A_min[i] to the table
                for i in range(p):
                    table_A.add_row(["%10.5e" % A[i], "%10.5e" % A_min[i]])

                # Print the table for A and A_min
                print(table_A)
        return A_min, chi2_min, b2_inverse_min

    def calibration(self):
        # Set the degree of the polynomial
        p = self.p
        # Extract data for reference ions only
        T = [ion.revolution_time for ion in self.data.ions if ion.is_reference_ion]
        MoQ = [ion.mass_Q for ion in self.data.ions if ion.is_reference_ion]
        TError = [ion.revolution_time_error for ion in self.data.ions if ion.is_reference_ion]
        MoQError = [ion.mass_Q_error for ion in self.data.ions if ion.is_reference_ion]

        # Print the extracted data
        fitting_data = []
        for ion in self.data.ions:
            if ion.is_reference_ion:
                ion_name = f"{ion.A_ion}{ion.element}{ion.Q_ion}+"
                fitting_data.append([ion_name, ion.revolution_time, ion.mass_Q, ion.revolution_time_error, ion.mass_Q_error])
        table = PrettyTable()
        table.field_names = ["Ion", "T(ps)", "MoQ", "TError", "MoQError"]
        for row in fitting_data:
            table.add_row(row)
        print(table)

        # Get initial parameters from ROOT fit
        self.initial_params, self.initial_covariance_matrix = self.get_initial_params_from_root(T, MoQ)

        # Calculate chi_squared
        self.initial_chi2 = sum((MoQ[i] - sum(self.initial_params[j] * mp.mpf(T[i]) ** j for j in range(p + 1))) ** 2 / MoQError[i] ** 2 for i in range(len(T)))

        self.params, self.chi2, self.covariance_matrix = self.LeastSquareFit(T, TError, MoQ, MoQError, self.p+1, self.iterationMax, self.initial_params, self.initial_covariance_matrix, self.OutputMatrixOrNot)

        for ion in self.data.ions:
            if not ion.is_reference_ion:
                # Calculate exp_mass_Q and exp_mass_Q_error for the temporarily excluded ion
                t = ion.revolution_time
                exp_mq = sum(self.params[i] * t ** i for i in range(self.p + 1))

                # Calculate average fitting error (sigma_fitav)
                sigma_fitav = mp.sqrt(sum(self.covariance_matrix[k, l] * (t ** k) * (t ** l) for k in range(self.p + 1) for l in range(self.p + 1)))

                # Calculate frequency error (sigma_freq)
                sigma_freq = mp.sqrt(sum((self.params[k] * (t ** (k - 1)) * mp.mpf(ion.revolution_time_error)) ** 2 for k in range(1, self.p + 1)))

                # Calculate total statistical error (sigma_stat)
                sigma_stat = mp.sqrt(sigma_fitav ** 2 + sigma_freq ** 2)

                ion.exp_mass_Q = exp_mq
                ion.exp_mass_Q_error = sigma_stat

                # Calculate mass and mass excess
                Mass_N = ion.exp_mass_Q * ion.Q_ion * amu
                ion.exp_mass_excess = Mass_N - ion.A_ion * amu + ion.Q_ion * me - ion.binding_energy
                ion.exp_mass_excess_error     = sigma_stat / ion.exp_mass_Q * Mass_N
                ion.exp_mass_excess_error_fitav = sigma_fitav / ion.exp_mass_Q * Mass_N
                ion.exp_mass_excess_error_freq  = sigma_freq / ion.exp_mass_Q * Mass_N

                # Store the beta parameters for the ion
                ion.initial_params = self.initial_params
                ion.initial_covariance_matrix = self.initial_covariance_matrix
                ion.params = self.params
                ion.covariance_matrix = self.covariance_matrix
                ion.chi2 = self.chi2
                print(f"Calibration for Ion: {ion.element}-{ion.A_ion}, chi2_min = {float(self.chi2):.5e}.")
                print("Fitting parameters are: ")
                for i in range(0, self.p+1):
                        print("%20.6e" % self.params[i])
                if isinstance(self.covariance_matrix, mp.matrix):
                    # Convert mpmath matrix to a list of lists
                    covariance_matrix_as_list = [[float(self.covariance_matrix[i, j]) for j in range(self.covariance_matrix.cols)] for i in range(self.covariance_matrix.rows)]

                    # Format the output
                    formatted_output = "\n".join(" ".join("%10.5e" % value for value in row) for row in covariance_matrix_as_list)
                    print(f"covariance_matrix:\n{formatted_output}")
                else:
                    print("Invalid covariance matrix format.")

                ion_data = [
                    ["sigma_fitav", f"{float(sigma_fitav):.5e}"],
                    ["sigma_freq", f"{float(sigma_freq):.5e}"],
                    ["sigma_stat", f"{float(sigma_stat):.5e}"],
                    ["m/q(exp)", f"{float(ion.exp_mass_Q):.5e}"],
                    ["m/q_error(exp) ", f"{float(ion.exp_mass_Q_error):.5e}"],
                    ["Mass_N (MeV)", f"{float(Mass_N):.5e}"],
                    ["Exp. Mass Excess (MeV)", f"{float(ion.exp_mass_excess):.5e}"],
                    ["Mass Excess(exp-ame) (keV)", f"{(float(ion.exp_mass_excess)-ion.mass_excess)*1000:.5e}"],
                    ["Exp. Mass Excess Total Error (keV)", f"{float(ion.exp_mass_excess_error) * 1000:.5e}"],
                    ["Exp. Mass Excess Fitting Error (keV)", f"{float(ion.exp_mass_excess_error_fitav) * 1000:.5e}"],
                    ["Exp. Mass Excess_Freq. Error (keV)", f"{float(ion.exp_mass_excess_error_freq) * 1000:.5e}"]
                ]
                ion_table = PrettyTable()
                ion_table.field_names = ["Parameter", "Value"]
                for row in ion_data:
                    ion_table.add_row(row)
                print(ion_table)

        
    def self_calibration(self):
        # Extract reference ions
        reference_ions = [ion for ion in self.data.ions if ion.is_reference_ion]
        
        # Self-calibration loop
        for ion in reference_ions:
            # Temporarily set the ion as non-reference
            ion.is_reference_ion = False
            print(f"\n\n # Self-Calibration for Ion: {ion.element}-{ion.A_ion}({ion.revolution_time} ps)")
            # Perform fit with remaining reference ions
            self.calibration()
            
            # Restore the ion's reference status
            ion.is_reference_ion = True
        
        
    def calculate_error_propagation(self, x, coeffs, x_error, p):
        """
        Calculate error propagation, compute the error in f based on the formula
        """
        f_error = mp.mpf(0)
        for k in range(1, p + 1):
            f_error += k * coeffs[k] * (x ** (k - 1)) * x_error
        return f_error

    def svd_inverse(self, A):
        """
        Compute the pseudo-inverse of matrix A using SVD
        """
        U, s, Vt = np.linalg.svd(A)
        s_inv = np.array([1 / si if si > 1e-10 else 0 for si in s])
        A_inv = Vt.T @ np.diag(s_inv) @ U.T
        return A_inv
