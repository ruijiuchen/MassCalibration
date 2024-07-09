import mpmath as mp
import numpy as np
import ROOT
from .constants import amu, me

class Model:
    def __init__(self, data, p,iterationMax,OutputMatrixOrNot,initial_params):
        self.data = data
        self.p = p
        self.iterationMax = iterationMax
        self.OutputMatrixOrNot = OutputMatrixOrNot
        self.initial_params = initial_params
        self.initial_covarianceMatrix = None
        self.initial_chi2 = None
        self.params = None
        self.covarianceMatrix = None
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
        covarianceMatrix = fitResult.GetCovarianceMatrix()
        
        return initial_params,covarianceMatrix

    def LeastSquareFit(self, T, TError, MoQ, MoQError, p,iterationMax,A0,OutputMatrixOrNot):
        N        =len(T)
        A        =A0[:]
        chi2_min =1e20
        A_min    =A[:]
        b2_inverse_min =[]
        iteration=0

        for iteration in range(0, iterationMax):
            FDB=[]
            PDB=[]
            # step 1: F matrix.
            for i in range(0, N):
                f   = T[i]
                row = []
                for k in range(0, p):
                    row.insert(k,f**k)
                FDB.append(row)
            # step 2 delta_MoQ_fi
            delta_MoQ_fiDB=[]
            for i in range(0, N):
                fi           = T[i]
                delta_fi     = TError[i]
                delta_MoQ_fi = 0
                for k in range(1, p):
                    delta_MoQ_fi = delta_MoQ_fi + k*A[k]*fi**(k-1)*delta_fi
                delta_MoQ_fiDB.insert(i,delta_MoQ_fi)
            #step 3 P matrix
            for i in range(0, N):
                row = []
                for j in range(0, N):
                    a=0
                    if i ==j:
                        a = 1.0/(float(MoQError[i])**2 + float(delta_MoQ_fiDB[i])**2)
                    row.insert(j, a)
                PDB.insert(i,row)
            # step 4 chi2
            if iteration==0:
                chi2 = 0
                for i in range(0, N):
                    fi        = T[i]
                    delta_fi  = TError[i]
                    y         = MoQ[i]
                    ye        = MoQError[i]
                    yfit      = 0
                    yfit_error=delta_MoQ_fiDB[i]
                    for k in range(0, p):
                        yfit = yfit + A[k]*fi**(k)
                    chi2 = chi2 + (y - yfit)**2 / (ye**2  + yfit_error**2)
                chi2_min=chi2
                A_min   =A[:]   # don't changed this line!   
                if OutputMatrixOrNot: print("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)

            # step 5 calculated parameters A
            F      = np.array(FDB)  # Convert the list of lists to a 2D NumPy array
            P      = np.array(PDB)  # Convert the list of lists to a 2D NumPy array
            F_T    = F.transpose() # Calculate the transpose of F
            a1     = np.dot(P, MoQ)
            a2     = np.dot(F_T, a1)
            b1     = np.dot(P, F)
            b2     = np.dot(F_T, b1)
            det_b2 = np.linalg.det(b2)
            chi2 = 0        
            if det_b2 == 0:
                if OutputMatrixOrNot: print("b2 = FT*P*F is a singular matrix. Iteration stop.")
                break
            else:
                b2_inverse= np.linalg.inv(b2)
                A         = np.dot(b2_inverse, a2)
                if iteration == 0: b2_inverse_min  = b2_inverse[:]
                # step 6
                for i in range(0, N):
                    fi        = T[i]
                    delta_fi  = TError[i]
                    y         = MoQ[i]
                    ye        = MoQError[i]
                    yfit      = 0
                    yfit_error=delta_MoQ_fiDB[i]
                    for k in range(0, p):
                        yfit = yfit + A[k]*fi**(k)
                    chi2 = chi2 + (y - yfit)**2 / (ye**2  + yfit_error**2)
                if chi2_min>=chi2:
                    chi2_min=chi2
                    A_min   =A[:]
                    b2_inverse_min  = b2_inverse[:]
                    if OutputMatrixOrNot: print("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)
                    iteration = iteration+1
            # step 7
            #############################plot matrix#############################################
            if OutputMatrixOrNot:
                if iteration < 4:
                    print("1.A=")
                    for k in range(0, p):
                        print("%10.3e"%A[k])

                    print("2.MoQDB +/- MoQDBError +/- delta_MoQ_fi | T +/- TError")
                    for i in range(0, N):
                        print("%10.5f"%MoQ[i]," +/- %10.5e"%MoQError[i]," +/- %10.5e"%delta_MoQ_fiDB[i], " | %10.5f"%T[i]," +/- %10.5f"%TError[i])

                    print("3.F=")
                    for i in range(0, N):
                        for k in range(0, p):
                            print("%10.3f"%F[i][k],end="\t")
                        print()
                    print("3.b F_T=")
                    for i in range(0, p):
                        for k in range(0, N):
                            print("%5.3e"%F_T[i][k],end="\t")
                        print()
                    print("4. P=")
                    for i in range(0, N):
                        for k in range(0, N):
                            if i == k:
                                print("%6.3f"%PDB[i][k],end="\t")
                            else:
                                print("%1d"%PDB[i][k],end="\t")
                        print()
                    print("5.a1 = P*MoQ")
                    for i in range(0, N):
                        print("%10.5f"%a1[i])
                    print("6.a2 = F_T*P*MoQ)")
                    for i in range(0, p):
                        print("%10.5f"%a2[i])
                    print("7. b1=P*F")
                    for i in range(0, N):
                        for k in range(0, p):
                            print("%6.3f"%b1[i][k],end="\t")
                        print()
                    print("8. b2=F_T*P*F")
                    for i in range(0, p):
                        for k in range(0, p):
                            print("%30.3f"%b2[i][k],end="\t")
                        print()
                    print("9. b2_inverse=(F_T*P*F)^-1")
                    for i in range(0, p):
                        for k in range(0, p):
                            print("%30.3e"%b2_inverse[i][k],end="\t")
                        print()
                    print("10.A=")
                    for i in range(0, p):
                        print("%10.5e"%A[i])
                    print("chi2=",chi2)
            if OutputMatrixOrNot: print("### iteration = ",iteration," chi2_min=","%10.5e"%chi2_min," A_min = ",A_min," chi2=","%10.5e"%chi2," A = ",A)
            
        #if OutputMatrixOrNot: print("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)
        return A_min, chi2_min,b2_inverse_min

    def calibration(self):
        # Set the degree of the polynomial
        p = self.p
        # Extract data for reference ions only
        T = [ion.revolution_time for ion in self.data.ions if ion.is_reference_ion]
        MoQ = [ion.mass_Q for ion in self.data.ions if ion.is_reference_ion]
        TError = [ion.revolution_time_error for ion in self.data.ions if ion.is_reference_ion]
        MoQError = [ion.mass_Q_error for ion in self.data.ions if ion.is_reference_ion]
        
        # Print the extracted data
        print("Fitting Data:")
        for ion in self.data.ions:
            if ion.is_reference_ion:
                ion_name = f"{ion.A_ion}{ion.element}{ion.Q_ion}+"
                t = ion.revolution_time
                moq = ion.mass_Q
                terr = ion.revolution_time_error
                moqerr = ion.mass_Q_error
                print(f"Ion: {ion_name}, T: {t}, MoQ: {moq}, TError: {terr}, MoQError: {moqerr}")
        
        # Get initial parameters from ROOT fit
        self.initial_params,self.initial_covarianceMatrix = self.get_initial_params_from_root(T, MoQ)
        
        # Calculate chi_squared
        self.initial_chi2 = sum((MoQ[i] - sum(self.initial_params[j] * mp.mpf(T[i]) ** j for j in range(p + 1))) ** 2 / MoQError[i] ** 2 for i in range(len(T)))
        
        self.params, self.chi2, self.covarianceMatrix = self.LeastSquareFit(T, TError, MoQ, MoQError, self.p+1, self.iterationMax ,self.initial_params , self.OutputMatrixOrNot)
        
        print(f"Initial parameters from ROOT fit: {self.initial_params}")
        print(f"Chi_squared with initial parameters: {self.initial_chi2}")
        
        for ion in self.data.ions:
            if ion.is_reference_ion == False:
                # Calculate exp_mass_Q and exp_mass_Q_error for the temporarily excluded ion
                t = ion.revolution_time
                exp_mq = sum(self.params[i] * t ** i for i in range(self.p + 1))

                # Calculate average fitting error (sigma_fitav)
                sigma_fitav = mp.sqrt(sum(self.covarianceMatrix[k, l] * (t ** k) * (t ** l) for k in range(self.p + 1) for l in range(self.p + 1)))

                # Calculate frequency error (sigma_freq)
                sigma_freq = mp.sqrt(sum((self.params[k] * (t ** (k - 1)) * mp.mpf(ion.revolution_time_error)) ** 2 for k in range(1, self.p + 1)))

                # Calculate total statistical error (sigma_stat)
                sigma_stat = mp.sqrt(sigma_fitav ** 2 + sigma_freq ** 2)

                ion.exp_mass_Q = exp_mq
                ion.exp_mass_Q_error = sigma_stat

                # Calculate mass and mass excess
                Mass_N = ion.exp_mass_Q * ion.Q_ion * amu
                ion.exp_mass_excess = Mass_N - ion.A_ion * amu + ion.Q_ion * me - ion.binding_energy
                ion.exp_mass_excess_error = ion.exp_mass_Q_error / ion.exp_mass_Q * Mass_N

                # Store the beta parameters for the ion
                ion.initial_params = self.initial_params
                ion.initial_covarianceMatrix = self.initial_covarianceMatrix
                ion.params = self.params
                print(f"Self-Calibration for Ion: {ion.element}-{ion.A_ion}")
                print(f"self.covarianceMatrix:\n{self.covarianceMatrix}")
                print(f"sigma_fitav: {sigma_fitav}")
                print(f"sigma_freq: {sigma_freq}")
                print(f"sigma_stat: {sigma_stat}")
                print(f"Exp_Mass_Q: {ion.exp_mass_Q}")
                print(f"Exp_Mass_Q_Error: {ion.exp_mass_Q_error}")
                print(f"Mass_N: {Mass_N} MeV")
                print(f"Exp_Mass_Excess: {ion.exp_mass_excess} MeV")
                print(f"Exp_Mass_Excess_Error: {ion.exp_mass_excess_error*1000} keV")            
            
        
    def self_calibration(self):
        # Extract reference ions
        reference_ions = [ion for ion in self.data.ions if ion.is_reference_ion]

        # Self-calibration loop
        for ion in reference_ions:
            # Temporarily set the ion as non-reference
            ion.is_reference_ion = False

            # Perform fit with remaining reference ions
            self.calibration()
            
            # Calculate exp_mass_Q and exp_mass_Q_error for the temporarily excluded ion
            t = mp.mpf(ion.revolution_time)
            exp_mq = sum(self.params[i] * t ** i for i in range(self.p + 1))
            
            # Calculate average fitting error (sigma_fitav)
            sigma_fitav = mp.sqrt(sum(self.covarianceMatrix[k, l] * (t ** k) * (t ** l) for k in range(self.p + 1) for l in range(self.p + 1)))

            # Calculate frequency error (sigma_freq)
            sigma_freq = mp.sqrt(sum((self.params[k] * (t ** (k - 1)) * mp.mpf(ion.revolution_time_error)) ** 2 for k in range(1, self.p + 1)))

            # Calculate total statistical error (sigma_stat)
            sigma_stat = mp.sqrt(sigma_fitav ** 2 + sigma_freq ** 2)
            
            ion.exp_mass_Q = exp_mq
            ion.exp_mass_Q_error = sigma_stat

            # Calculate mass and mass excess
            Mass_N = ion.exp_mass_Q * ion.Q_ion * amu
            ion.exp_mass_excess = Mass_N - ion.A_ion * amu + ion.Q_ion * me - ion.binding_energy
            ion.exp_mass_excess_error = ion.exp_mass_Q_error / ion.exp_mass_Q * Mass_N

            # Store the beta parameters for the ion
            ion.initial_params = self.initial_params
            ion.initial_covarianceMatrix = self.initial_covarianceMatrix
            ion.beta = self.params
            
            # Print intermediate variables for debugging
            print(f"Self-Calibration for Ion: {ion.element}-{ion.A_ion}")
            print(f"self.covarianceMatrix:\n{self.covarianceMatrix}")
            print(f"sigma_fitav: {sigma_fitav}")
            print(f"sigma_freq: {sigma_freq}")
            print(f"sigma_stat: {sigma_stat}")
            print(f"Exp_Mass_Q: {ion.exp_mass_Q}")
            print(f"Exp_Mass_Q_Error: {ion.exp_mass_Q_error}")
            print(f"Mass_N: {Mass_N}")
            print(f"Exp_Mass_Excess: {ion.exp_mass_excess}")
            print(f"Exp_Mass_Excess_Error: {ion.exp_mass_excess_error}")
            
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
