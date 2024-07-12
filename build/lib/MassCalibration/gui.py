import mpmath as mp
import numpy as np
import ROOT

class GUI:
    def __init__(self, data, model):
        self.data = data
        self.model = model
    
    def draw_individual_fits(self):
        # Draw the graph with experimental data for all ions
        graph = ROOT.TGraphErrors()
        graph.SetTitle("Fit for All Ions; T; MoQ")
        graph.SetMarkerStyle(21)
        graph.SetMarkerSize(1)
        graph.SetMarkerColor(ROOT.kBlue)
        i = 0
        for ion in self.data.ions:
            graph.SetPoint(i, ion.revolution_time, ion.mass_Q)
            graph.SetPointError(i, ion.revolution_time_error, ion.mass_Q_error)
            i += 1

        # Find the min and max revolution time
        min_revolution_time = min(ion.revolution_time for ion in self.data.ions)
        max_revolution_time = max(ion.revolution_time for ion in self.data.ions)

        for ion in self.data.ions:
            canvas = ROOT.TCanvas(f"canvas_{ion.element}_{ion.A_ion}", f"Fit for {ion.A_ion}{ion.element}", 800, 600)
            canvas.Divide(2, 2)
            # Draw the experimental data
            canvas.cd(1)
            graph.Draw("AP")

            if np.any(ion.params):
                # Create a TF1 object using the params parameters from the ion
                func = ROOT.TF1(f"func_{ion.element}_{ion.A_ion}", f"pol{self.data.p}", min_revolution_time, max_revolution_time)
                for i in range(self.data.p + 1):
                    func.SetParameter(i, float(ion.params[i])) 
                func.SetLineColor(ROOT.kGreen)

                # Draw the fitting function
                func.Draw("same")

                # Print beta values on the canvas
                beta_text = ROOT.TLatex()
                beta_text.SetNDC()
                beta_text.SetTextSize(0.03)
                beta_text.DrawLatex(0.15, 0.85, f"params: {ion.params}")
                
                canvas.cd(2)
                graph.Draw("AP")

                # Draw the residuals
                canvas.cd(3)
                residuals = ROOT.TGraphErrors()
                chi2 = 0
                for j, ion_residual in enumerate(self.data.ions):
                    residual = ion_residual.mass_Q - func.Eval(ion_residual.revolution_time)
                    error_sum = ion_residual.mass_Q_error ** 2 + ion_residual.exp_mass_Q_error ** 2
                    chi2 += (residual ** 2) / error_sum
                    residuals.SetPoint(j, ion_residual.revolution_time, residual)
                    residuals.SetPointError(j, ion_residual.revolution_time_error, ion_residual.mass_Q_error)
                residuals.SetTitle(f"Residuals for All Ions with {ion.A_ion}{ion.element} Fit; T; Residuals")
                residuals.SetMarkerStyle(21)
                residuals.SetMarkerSize(1)
                residuals.SetMarkerColor(ROOT.kRed)
                residuals.Draw("AP")
                
                canvas.cd(4)
                for i in range(self.data.p + 1):
                    func.SetParameter(i, float(ion.initial_params[i])) 
                residuals_initial_params = ROOT.TGraphErrors()
                chi2 = 0
                for j, ion_residual in enumerate(self.data.ions):
                    residual = ion_residual.mass_Q - func.Eval(ion_residual.revolution_time)
                    error_sum = ion_residual.mass_Q_error ** 2 + ion_residual.exp_mass_Q_error ** 2
                    chi2 += (residual ** 2) / error_sum
                    residuals_initial_params.SetPoint(j, ion_residual.revolution_time, residual)
                    residuals_initial_params.SetPointError(j, ion_residual.revolution_time_error, ion_residual.mass_Q_error)
                residuals_initial_params.SetTitle(f"Residuals for All Ions with {ion.A_ion}{ion.element} Fit; T; Residuals")
                residuals_initial_params.SetMarkerStyle(21)
                residuals_initial_params.SetMarkerSize(1)
                residuals_initial_params.SetMarkerColor(ROOT.kRed)
                residuals_initial_params.Draw("AP")

                # Print chi2 value on the canvas
                chi2_text = ROOT.TLatex()
                chi2_text.SetNDC()
                chi2_text.SetTextSize(0.03)
                chi2_text.DrawLatex(0.15, 0.85, f"Chi2: {chi2}")

            canvas.Update()
            canvas.SaveAs(f"Fit_{ion.A_ion}{ion.element}.png")
            
    def draw_me_vs_tof(self, label_offset=5, x_min=None, x_max=None, y_min=None, y_max=None):
        # Create TGraphErrors for ME(EXP-AME) vs TOF
        graph_me_exp_ame = ROOT.TGraphErrors()
        graph_me_exp_ame.SetTitle("ME(EXP-AME) vs TOF; TOF (ns); ME(EXP-AME) (keV)")
        graph_me_exp_ame.SetMarkerStyle(20)
        graph_me_exp_ame.SetMarkerSize(1)
        graph_me_exp_ame.SetMarkerColor(ROOT.kBlack)

        # Create TGraphErrors for mass_excess vs TOF
        graph_me = ROOT.TGraphErrors()
        graph_me.SetTitle(f" #chi^{{2}} = {float(self.data.chi2_me):.2f}, #sigma_{{syst}} = {float(self.data.sigma_syst_me * 1000):.2f} keV")
        graph_me.SetFillColor(7)
        graph_me.SetLineColor(7)
        graph_me.SetLineWidth(2)
        graph_me.SetMarkerColor(7)

        # Create TGraphErrors for ME(EXP-AME) vs TOF for non-reference ions
        graph_me_exp_ame_non_ref = ROOT.TGraphErrors()
        graph_me_exp_ame_non_ref.SetTitle("ME(EXP-AME) vs TOF (Non-reference Ions); TOF (ns); ME(EXP-AME) (keV)")
        graph_me_exp_ame_non_ref.SetMarkerStyle(21)  # Square markers
        graph_me_exp_ame_non_ref.SetMarkerSize(1)
        graph_me_exp_ame_non_ref.SetMarkerColor(ROOT.kRed)
        graph_me_exp_ame_non_ref.SetLineColor(ROOT.kRed)

        i = 0
        j = 0
        k = 0  # New index for non-reference ions
        for ion in self.data.ions:
            if ion.exp_mass_excess is not None and ion.mass_excess is not None:
                tof = ion.revolution_time / 1000
                me_exp_ame = (ion.exp_mass_excess - ion.mass_excess) * 1000  # Convert to keV
                tof_error = ion.revolution_time_error / 1000
                me_error = ion.exp_mass_excess_error * 1000  # Convert to keV

                graph_me_exp_ame.SetPoint(i, tof, me_exp_ame)
                graph_me_exp_ame.SetPointError(i, tof_error, me_error)
                i += 1

                graph_me.SetPoint(j, tof, 0)
                graph_me.SetPointError(j, tof_error, ion.mass_excess_error * 1000)
                j += 1

                if not ion.is_reference_ion:
                    graph_me_exp_ame_non_ref.SetPoint(k, tof, me_exp_ame)
                    graph_me_exp_ame_non_ref.SetPointError(k, tof_error, me_error)
                    k += 1

        canvas = ROOT.TCanvas("canvas_me_vs_tof", "canvas_me_vs_tof", 910, 700)
        canvas.SetFillColor(0)
        canvas.SetBorderMode(0)
        canvas.SetBorderSize(2)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)  # Adjusted margin to ensure x-axis title is not blocked
        canvas.SetFrameBorderMode(0)

        # Draw ME(EXP-AME) vs TOF
        graph_me.Draw("A3")
        graph_me.Draw("sameP")
        graph_me_exp_ame.Draw("sameP")
        graph_me_exp_ame_non_ref.Draw("sameP")

        # Set axis parameters with optional input values
        if x_min is None:
            x_min = graph_me_exp_ame.GetXaxis().GetXmin()
        if x_max is None:
            x_max = graph_me_exp_ame.GetXaxis().GetXmax()
        if y_min is None:
            y_min = graph_me_exp_ame.GetYaxis().GetXmin()
        if y_max is None:
            y_max = graph_me_exp_ame.GetYaxis().GetXmax()

        graph_me.GetXaxis().SetTitle("Revolution time (ns)")
        graph_me.GetXaxis().SetLimits(x_min, x_max)
        graph_me.GetXaxis().SetNdivisions(505)
        graph_me.GetXaxis().CenterTitle(True)
        graph_me.GetXaxis().SetLabelFont(42)
        graph_me.GetXaxis().SetLabelSize(0.06)
        graph_me.GetXaxis().SetTitleSize(0.06)
        graph_me.GetXaxis().SetTitleFont(42)
        graph_me.GetYaxis().SetTitle("ME-AME16(keV)")
        graph_me.GetYaxis().CenterTitle(True)
        graph_me.GetYaxis().SetNdivisions(510)
        graph_me.GetYaxis().SetLabelFont(42)
        graph_me.GetYaxis().SetLabelSize(0.06)
        graph_me.GetYaxis().SetTitleSize(0.06)
        graph_me.GetYaxis().SetTitleFont(42)
        graph_me.GetYaxis().SetTitleOffset(1.2)
        graph_me.GetYaxis().SetRangeUser(y_min, y_max)

        line = ROOT.TLine(x_min, 0, x_max, 0)
        line.SetLineStyle(2)  # Set line style to dashed
        line.Draw("same")

        # Add labels for each ion
        label = ROOT.TLatex()
        label.SetTextSize(0.03)
        label.SetTextAlign(22)  # Align center

        for ion in self.data.ions:
            if ion.exp_mass_excess is not None and ion.mass_excess is not None:
                tof = ion.revolution_time / 1000
                me_exp_ame = (ion.exp_mass_excess - ion.mass_excess) * 1000  # Convert to keV
                me_error = ion.exp_mass_excess_error * 1000  # Convert to keV
                ion_name = f"^{{{ion.A_ion}}}{ion.element}_{{{ion.Z_ion}}}^{{{ion.Q_ion}+}}"
                label.SetTextColor(ROOT.kMagenta if ion.Z_ion != ion.Q_ion else ROOT.kBlack)
                label.DrawLatex(tof, me_exp_ame + me_error + label_offset, ion_name)  # Position label above error bar

        # Add legend
        legend = ROOT.TLegend(0.15, 0.85, 0.85, 0.92)  # Position the legend outside the frame
        legend.SetNColumns(2)  # Set number of columns in the legend
        legend.AddEntry(graph_me_exp_ame, "Reference Ions", "p")
        legend.AddEntry(graph_me_exp_ame_non_ref, "Non-reference Ions", "p")
        legend.Draw()

        # Update and save the canvas
        canvas.Update()
        
        canvas.SaveAs("ME_vs_TOF_with_labels.png")
        canvas.SaveAs("ME_vs_TOF_with_labels.root")
        return canvas


