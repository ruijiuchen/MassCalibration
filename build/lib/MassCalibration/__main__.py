import sys
import toml
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QHBoxLayout, QMainWindow, QFrame, QComboBox, QTextEdit, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ROOT import TCanvas, TGraphErrors, TLatex, TLegend, TLine, kBlack, kRed, kMagenta

from MassCalibration.ion import Ion
from MassCalibration.data import Data
from MassCalibration.model import Model
from MassCalibration.gui import GUI

class CalibrationApp(QWidget):
    def __init__(self, config_file=None):
        super().__init__()
        self.config_file = config_file or os.path.join(os.getenv('MASSCALIBRATION_ROOT', ''), 'config.toml')
        self.initUI()
        
    def initUI(self):
        self.layout = QVBoxLayout()
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Load config button
        self.loadButton = QPushButton('Load Config', self)
        self.loadButton.setStyleSheet("background-color: lightblue")
        self.loadButton.clicked.connect(self.loadConfig)
        button_layout.addWidget(self.loadButton)

        # Save config button
        self.saveButton = QPushButton('Save Config', self)
        self.saveButton.setStyleSheet("background-color: lightyellow")
        self.saveButton.clicked.connect(self.saveConfig)
        button_layout.addWidget(self.saveButton)
        
        # Run button
        self.runButton = QPushButton('Run', self)
        self.runButton.setStyleSheet("background-color: lightgreen")
        self.runButton.clicked.connect(self.runCalibration)
        button_layout.addWidget(self.runButton)
        
        self.layout.addLayout(button_layout)
        
        # Config parameters
        self.params = {}
        self.param_labels = {}
        self.param_inputs = {}
        self.param_browsers = {}
        self.param_editors = {}
        
        self.config = None
        if self.config_file and os.path.exists(self.config_file):
            self.load_config_from_file(self.config_file)
        
        self.setLayout(self.layout)
        self.setWindowTitle('Mass Calibration')
        self.show()
    
    def loadConfig(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        config_file, _ = QFileDialog.getOpenFileName(self, "Load Config File", "", "TOML Files (*.toml);;All Files (*)", options=options)
        if config_file:
            self.load_config_from_file(config_file)
            
    def load_config_from_file(self, config_file):
        self.config = toml.load(config_file)
        self.displayConfig()

    def saveConfig(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        config_file, _ = QFileDialog.getSaveFileName(self, "Save Config File", "", "TOML Files (*.toml);;All Files (*)", options=options)
        if config_file:
            self.save_config_to_file(config_file)
    
    def save_config_to_file(self, config_file):
        parameters = {key: self.param_inputs[key].currentText() if key in ['draw_individual_fits', 'OutputMatrixOrNot'] else self.param_inputs[key].text() for key in self.param_inputs}
        with open(config_file, 'w') as f:
            toml.dump({'parameters': parameters}, f)
    
    def save_config_to_default_file(self):
        self.save_config_to_file(self.config_file)

    def displayConfig(self):
        if not self.config:
            return
        
        # Update existing parameter inputs or create new ones if they do not exist
        for key, value in self.config['parameters'].items():
            if key in self.param_inputs:
                if key in ['draw_individual_fits', 'OutputMatrixOrNot']:
                    self.param_inputs[key].setCurrentText(str(value))
                else:
                    self.param_inputs[key].setText(str(value))
            else:
                if key in ['ame_file', 'elbien_file', 'revtime_file']:
                    self.create_file_input(key, value)
                elif key in ['draw_individual_fits', 'OutputMatrixOrNot']:
                    self.create_combobox_input(key, value)
                else:
                    label = QLabel(f"{key}:")
                    self.layout.addWidget(label)
                    self.param_labels[key] = label

                    input_field = QLineEdit(str(value))
                    self.layout.addWidget(input_field)
                    self.param_inputs[key] = input_field
            
    def create_file_input(self, key, value):
        h_layout = QHBoxLayout()
        
        label = QLabel(f"{key}:")
        h_layout.addWidget(label)
        self.param_labels[key] = label

        input_field = QLineEdit(str(value))
        h_layout.addWidget(input_field)
        self.param_inputs[key] = input_field
        
        browse_button = QPushButton('Browse', self)
        browse_button.clicked.connect(lambda: self.browse_file(key))
        h_layout.addWidget(browse_button)
        self.param_browsers[key] = browse_button

        edit_button = QPushButton('Edit', self)
        edit_button.clicked.connect(lambda: self.edit_file(key))
        h_layout.addWidget(edit_button)
        self.param_editors[key] = edit_button
        
        self.layout.addLayout(h_layout)

    def create_combobox_input(self, key, value):
        h_layout = QHBoxLayout()
        
        label = QLabel(f"{key}:")
        h_layout.addWidget(label)
        self.param_labels[key] = label

        combobox = QComboBox()
        combobox.addItems(['true', 'false'])
        combobox.setCurrentText(str(value))
        h_layout.addWidget(combobox)
        self.param_inputs[key] = combobox
        
        self.layout.addLayout(h_layout)

    def browse_file(self, key):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, f"Select {key}", "", "All Files (*)", options=options)
        if file_name:
            self.param_inputs[key].setText(file_name)

    def edit_file(self, key):
        file_path = self.param_inputs[key].text()
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", f"The file {file_path} does not exist.")
            return
        
        editor = QTextEdit()
        editor.setWindowTitle(f"Editing {key}")
        editor.setGeometry(100, 100, 800, 600)

        with open(file_path, 'r') as file:
            editor.setText(file.read())
        
        def save_changes():
            with open(file_path, 'w') as file:
                file.write(editor.toPlainText())
            editor.close()
            QMessageBox.information(self, "Saved", f"Changes to {key} have been saved.")
        
        save_button = QPushButton("Save", editor)
        save_button.clicked.connect(save_changes)
        save_button.setGeometry(700, 550, 80, 30)
        
        editor.show()

    def create_plot_window(self, image_path):
        self.plot_window = QMainWindow()
        self.plot_window.setWindowTitle("Plot")
        self.plot_window.setGeometry(100, 100, 800, 600)
        self.plot_frame = QFrame(self.plot_window)
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_window.setCentralWidget(self.plot_frame)
        
        # Display the image in the PyQt5 window
        label = QLabel(self.plot_frame)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        self.plot_layout.addWidget(label)
        
        self.plot_window.show()
    
    def runCalibration(self):
        if not self.config:
            return

        # Read values from input fields
        for key in self.param_inputs:
            if key in ['draw_individual_fits', 'OutputMatrixOrNot']:
                self.config['parameters'][key] = self.param_inputs[key].currentText()
            else:
                self.config['parameters'][key] = self.param_inputs[key].text()

        # Save the current configuration to the default config file
        self.save_config_to_default_file()

        # Convert parameters to appropriate types
        ame_file = self.config['parameters'].get('ame_file', 'default_ame_file.rd')
        elbien_file = self.config['parameters'].get('elbien_file', 'default_elbien_file.dat')
        revtime_file = self.config['parameters'].get('revtime_file', 'default_revtime_file.txt')
        p = int(self.config['parameters'].get('p', 2))
        iterationMax = int(self.config['parameters'].get('iterationMax', 100))
        OutputMatrixOrNot = self.config['parameters'].get('OutputMatrixOrNot', 'true') == 'true'
        initial_params = list(map(float, self.config['parameters'].get('initial_params', '[0, 0, 0]').strip('[]').split(',')))
        label_offset = int(self.config['parameters'].get('label_offset', 5))
        
        # Ensure y_min and y_max are properly set
        me_vs_tof_y_min = self.config['parameters'].get('me_vs_tof_y_min')
        me_vs_tof_y_max = self.config['parameters'].get('me_vs_tof_y_max')

        if me_vs_tof_y_min is not None:
            me_vs_tof_y_min = float(me_vs_tof_y_min)
        if me_vs_tof_y_max is not None:
            me_vs_tof_y_max = float(me_vs_tof_y_max)

        draw_individual_fits = self.config['parameters'].get('draw_individual_fits', 'true') == 'true'
        
        print(f"draw_individual_fits: {draw_individual_fits}")  # Debug print

        data = Data(ame_file, elbien_file, revtime_file, p)
        model = Model(data, p, iterationMax, OutputMatrixOrNot, initial_params)
        model.self_calibration()  # Perform the fit before drawing the graph
        model.calibration()  # Perform the fit before drawing the graph
        data.calculate_chi2_and_systematic_error(p)  # Calculate chi2 and systematic error
        data.print_final_experimental_me()  # Print final experimental ME and error for all ions
        gui = GUI(data, model)
        if draw_individual_fits:
            print("Running gui.draw_individual_fits()")  # Debug print
            gui.draw_individual_fits()
        
        # Draw the plot and save it as an image
        image_path = "ME_vs_TOF_with_labels.png"
        gui.draw_me_vs_tof(label_offset=label_offset, y_min=me_vs_tof_y_min, y_max=me_vs_tof_y_max)
        
        # Create a new window for the plot
        self.create_plot_window(image_path)

def main():
    # Set the default config file location
    default_config_file = os.path.join(os.getenv('MASSCALIBRATION_ROOT', ''), 'config.toml')
    
    # Check if a config file was provided as a command-line argument
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = default_config_file
    
    # Run the application
    app = QApplication(sys.argv)
    ex = CalibrationApp(config_file)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
