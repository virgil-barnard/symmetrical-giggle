import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QComboBox,
    QGridLayout, QLineEdit, QWidget, QMessageBox
)
from PyQt6.QtGui import QIntValidator
from collections import deque
import uhd
#from dsp_utils import DSPProcessor
class DSPProcessor:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def apply_lowpass_filter(self, samples, cutoff):
        # Implement a basic lowpass filter here
        return samples

class SDR:
    def set_frequency(self, val_hz):
        pass

    def set_sample_rate(self, sample_rate):
        pass

    def set_bandwidth(self, bandwidth):
        pass

    def get_samples(self):
        pass

class EttusN310SDR(SDR):
    def __init__(self, sample_rate=1e6, center_frequency=100e6, num_samples=10000, bandwidth=2e6, device_args=""):
        self.SAMPLE_RATE = sample_rate
        self.CENTER_FREQUENCY = center_frequency
        self.NUM_SAMPLES = num_samples
        self.BANDWIDTH = bandwidth
        self.sdr = uhd.usrp.MultiUSRP(device_args)
        self.sdr.set_rx_rate(self.SAMPLE_RATE)
        self.sdr.set_rx_freq(uhd.types.TuneRequest(self.CENTER_FREQUENCY))
        self.sdr.set_rx_bandwidth(self.BANDWIDTH)

    def set_sample_rate(self, sample_rate):
        self.SAMPLE_RATE = sample_rate
        self.sdr.set_rx_rate(sample_rate)
    
    def set_frequency(self, val_hz):
        self.CENTER_FREQUENCY = val_hz
        self.sdr.set_rx_freq(uhd.types.TuneRequest(val_hz))

    def set_bandwidth(self, bandwidth):
        self.BANDWIDTH = bandwidth
        self.sdr.set_rx_bandwidth(bandwidth)

    def get_samples(self):
        samples = self.sdr.recv_num_samps(self.NUM_SAMPLES, self.CENTER_FREQUENCY, self.SAMPLE_RATE, [0], 50)
        if samples is None:
            raise RuntimeError("Failed to receive samples.")
        return samples

class SpectrogramApp(QMainWindow):
    def __init__(self, sdr):
        super().__init__()
        self.sdr = sdr
        self.SAMPLE_RATE = self.sdr.SAMPLE_RATE
        self.CENTER_FREQUENCY = self.sdr.CENTER_FREQUENCY
        self.NUM_SAMPLES = self.sdr.NUM_SAMPLES
        self.MIN_FREQ_HZ = 10000000
        self.MAX_FREQ_HZ = 4000000000
        self.NUM_AVERAGES = 20
        self.NUM_SPECTROGRAM = 50
        self.PEAK_THRESHOLD = 25
        self.PEAK_DISTANCE = 20
        self.BANDWIDTH = 4e6
        self.highcut = self.sdr.BANDWIDTH // 2
        self.window_type = 'hamming'

        self.initUI()
        self.dsp = DSPProcessor(self.SAMPLE_RATE)
        self.spectrogram_data = deque(maxlen=self.NUM_SPECTROGRAM)
        self.average_data = deque(maxlen=self.NUM_AVERAGES)

    def initUI(self):
        self.setWindowTitle('Spectrogram')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QGridLayout()
        central_widget.setLayout(layout)

        self.waterfall = pg.GraphicsLayoutWidget()
        self.line_plot = self.waterfall.addPlot(row=0, col=0)
        self.line = self.line_plot.plot(pen='r')
        self.line_peaks = self.line_plot.plot(pen=None, symbol='o', symbolBrush='r', symbolSize=5)
        self.spectrogram_plot = self.waterfall.addPlot(row=1, col=0)
        self.img = pg.ImageItem()
        self.spectrogram_plot.addItem(self.img)
        self.img.setLookupTable(pg.colormap.get("CET-L5").getLookupTable())
        layout.addWidget(self.waterfall, 0, 0, 1, 12)

        self.colormaps = ['CET-C1', 'CET-D13', 'CET-C6', 'CET-D4', 'CET-I2', 'CET-D3', 'CET-CBL1', 'CET-L18',
                          'inferno', 'CET-R1', 'CET-L11', 'CET-C7s', 'CET-L16', 'CET-L4', 'PAL-relaxed', 
                          'CET-CBTL2', 'CET-L3', 'CET-CBD1', 'CET-C3s', 'CET-I3', 'CET-D2', 'CET-C6s', 
                          'CET-D12', 'CET-CBTC1', 'CET-C7', 'CET-C2s', 'CET-CBC2', 'CET-L2', 'PAL-relaxed_bright', 
                          'plasma', 'CET-L5', 'cividis', 'CET-L17', 'CET-L10', 'CET-L19', 'CET-CBC1', 
                          'magma', 'CET-L8', 'CET-L6', 'CET-C4s', 'CET-L1', 'CET-L13', 'CET-L14', 
                          'CET-R3', 'CET-R4', 'CET-D6', 'CET-D1', 'CET-C3', 'CET-D11', 'CET-D8', 
                          'CET-CBTC2', 'CET-C4', 'viridis', 'CET-R2', 'CET-L15', 'CET-D1A', 'CET-L12', 
                          'CET-L7', 'CET-CBTL1', 'CET-L9', 'CET-C5s', 'CET-D10', 'CET-D9', 'CET-C5', 
                          'CET-C2', 'CET-I1', 'CET-CBL2', 'CET-CBTD1', 'CET-D7', 'CET-C1s']
        self.colormap_selector = QComboBox()
        self.colormap_label = QLabel("Colormap:")
        self.colormap_selector.addItems(self.colormaps)
        self.colormap_selector.setCurrentIndex(self.colormaps.index('CET-L5'))
        self.colormap_selector.currentIndexChanged.connect(self.update_colormap)
        layout.addWidget(self.colormap_label, 3, 3)
        layout.addWidget(self.colormap_selector, 3, 4)

        self.line_colors = ['Red', 'Blue', 'Green', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
        self.line_color_label = QLabel("Line Color:")
        self.line_color_selector = QComboBox()
        self.line_color_selector.addItems(self.line_colors)
        self.line_color_selector.currentIndexChanged.connect(self.update_line_color)
        self.line_color_selector.setCurrentIndex(self.line_colors.index('Red'))
        layout.addWidget(self.line_color_label, 4, 3)
        layout.addWidget(self.line_color_selector, 4, 4)

        self.num_averages_label = QLabel("Num Averages:")
        self.num_averages_textbox = QLineEdit(str(self.NUM_AVERAGES))
        self.num_averages_textbox.setValidator(QIntValidator(1, 300))
        self.num_averages_textbox.returnPressed.connect(self.update_num_averages_from_textbox)
        layout.addWidget(self.num_averages_label, 6, 0)
        layout.addWidget(self.num_averages_textbox, 6, 1) 

        self.num_samples_label = QLabel("Num Samples:")
        self.num_samples_textbox = QLineEdit(str(self.NUM_SAMPLES))
        self.num_samples_textbox.setValidator(QIntValidator(1, int(1e6)))  # Allow integers between 1 and 1,000,000
        self.num_samples_textbox.returnPressed.connect(self.update_num_samples_from_textbox)
        layout.addWidget(self.num_samples_label, 2, 0)  
        layout.addWidget(self.num_samples_textbox, 2, 1)  

        self.sample_rate_label = QLabel("Sample Rate:")
        self.sample_rate_textbox = QLineEdit(str(self.SAMPLE_RATE))
        self.sample_rate_textbox.setValidator(QIntValidator(1, int(1e8)))  # Allow integers between 1 and 100,000,000
        self.sample_rate_textbox.returnPressed.connect(self.update_sample_rate_from_textbox)
        layout.addWidget(self.sample_rate_label, 3, 0)  
        layout.addWidget(self.sample_rate_textbox, 3, 1)   

        self.center_frequency_label = QLabel("Center Frequency (Hz):")
        self.center_frequency_textbox = QLineEdit(str(self.CENTER_FREQUENCY))
        self.center_frequency_textbox.returnPressed.connect(self.update_center_frequency_from_textbox)
        layout.addWidget(self.center_frequency_label, 1, 0)
        layout.addWidget(self.center_frequency_textbox, 1, 1)

        self.peak_threshold_label = QLabel("Peak Threshold (dB):")
        self.peak_threshold_textbox = QLineEdit(str(self.PEAK_THRESHOLD))
        self.peak_threshold_textbox.returnPressed.connect(self.update_peak_threshold_from_textbox)
        layout.addWidget(self.peak_threshold_label, 4, 0)
        layout.addWidget(self.peak_threshold_textbox, 4, 1)

        self.peak_distance_label = QLabel("Peak Distance:")
        self.peak_distance_textbox = QLineEdit(str(self.PEAK_DISTANCE))
        self.peak_distance_textbox.setValidator(QIntValidator(1, 500))
        self.peak_distance_textbox.returnPressed.connect(self.update_peak_distance_from_textbox)
        layout.addWidget(self.peak_distance_label, 5, 0)
        layout.addWidget(self.peak_distance_textbox, 5, 1)

        self.highcut_label = QLabel("Highcut Frequency (Hz):")
        self.highcut_textbox = QLineEdit(str(self.highcut))
        self.highcut_textbox.setValidator(QIntValidator(0, int(self.SAMPLE_RATE / 2)))
        self.highcut_textbox.returnPressed.connect(self.update_highcut_from_textbox)
        layout.addWidget(self.highcut_label, 1, 3)
        layout.addWidget(self.highcut_textbox, 1, 4)

        self.window_types = ['hamming', 'hann', 'blackman', 'flattop']
        self.window_type_label = QLabel("Window Type:")
        self.window_type_selector = QComboBox()
        self.window_type_selector.addItems(self.window_types)
        self.window_type_selector.currentIndexChanged.connect(self.update_window_type)
        layout.addWidget(self.window_type_label, 2, 3)
        layout.addWidget(self.window_type_selector, 2, 4)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_application)
        layout.addWidget(self.exit_button, 5, 4)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.run)
        layout.addWidget(self.start_button, 6, 4)

    def update_num_averages_from_textbox(self):
        value = int(self.num_averages_textbox.text().strip())
        self.NUM_AVERAGES = value
        self.average_data.clear()
        self.average_data = deque(maxlen=self.NUM_AVERAGES)

    def update_num_samples_from_textbox(self):
        value = int(self.num_samples_textbox.text())
        self.NUM_SAMPLES = value

        self.average_data.clear()
        for _ in range(self.NUM_AVERAGES):
            self.average_data.append(np.zeros(self.NUM_SAMPLES))

        self.spectrogram_data.clear()
        for _ in range(self.NUM_SPECTROGRAM):
            self.spectrogram_data.append(np.zeros(self.NUM_SAMPLES))

    def update_sample_rate_from_textbox(self):
        self.SAMPLE_RATE = int(self.sample_rate_textbox.text())
        self.dsp = DSPProcessor(self.SAMPLE_RATE)
        self.sdr.set_sample_rate(int(self.SAMPLE_RATE))

    def update_highcut_from_textbox(self):
        self.highcut = int(self.highcut_textbox.text())

    def update_window_type(self, index):
        self.window_type = self.window_types[index]

    def update_center_frequency_from_textbox(self):
        self.CENTER_FREQUENCY = int(self.center_frequency_textbox.text())
        self.update_frequency(self.CENTER_FREQUENCY)

    def update_peak_threshold_from_textbox(self):
        self.PEAK_THRESHOLD = int(self.peak_threshold_textbox.text())

    def update_peak_distance_from_textbox(self):
        self.PEAK_DISTANCE = int(self.peak_distance_textbox.text())

    def update_line_color(self, index):
        color_name = self.line_colors[index]
        self.line.setPen(color_name)

    def update_colormap(self, index):
        colormap_name = self.colormaps[index]
        self.img.setLookupTable(pg.colormap.get(colormap_name).getLookupTable())

    def update_frequency(self, val_hz):
        self.sdr.set_frequency(val_hz)
        self.center_frequency_label.setText(f"Center Frequency: {val_hz} MHz")
    
    def exit_application(self):
        QApplication.quit()

    def to_dB(self, magnitude_array):
        magnitude_array = np.abs(magnitude_array)
        return 20 * np.log10(magnitude_array / 2**12)

    def find_peaks(self, data):
        peaks = []
        length = len(data)
        distance = self.PEAK_DISTANCE
        threshold = self.PEAK_THRESHOLD
        for i in range(distance, length - distance):
            if data[i] > threshold:
                is_peak = True
                for j in range(1, distance + 1):
                    if data[i] <= data[i - j] or data[i] <= data[i + j]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
        return peaks

    def run(self):
        try:
            while True:
                QApplication.processEvents()
                iq_samples = self.sdr.get_samples()[0]
                #iq_samples = iq_samples[0] + iq_samples[1]
                filtered_samples = self.dsp.apply_lowpass_filter(iq_samples, self.highcut)
                magnitude_spectrum = self.to_dB(np.fft.fftshift(np.fft.fft(filtered_samples)))
                frequencies = np.linspace(-self.SAMPLE_RATE / 2, self.SAMPLE_RATE / 2, len(magnitude_spectrum))
                self.average_data.append(magnitude_spectrum)
                averaged_spectrum = np.mean(np.array(self.average_data), axis=0)
                peaks = self.find_peaks(averaged_spectrum)
                peak_values = averaged_spectrum[peaks]
                peak_frequencies = frequencies[peaks]
                self.line_peaks.setData(x=peak_frequencies, y=peak_values, symbol='o')
                self.line.setData(x=frequencies, y=averaged_spectrum)
                self.spectrogram_data.append(magnitude_spectrum)
                self.img.setImage(np.array(self.spectrogram_data).T)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sdr = EttusN310SDR()
    spectrogram_app = SpectrogramApp(sdr)
    spectrogram_app.show()
    sys.exit(app.exec())

