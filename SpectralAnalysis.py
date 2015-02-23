"""
Spectral Analysis
"""

import numpy
import math
import re

from matplotlib import pyplot as plt


class SpectralAnalysis(object):
    def __init__(self, reader):
        self.reader = reader  # contains data from fds file
        self.rdf = reader.file_in  # link to RawDataFile

        self.dist_unit = 'm'
        self.dist_rng = 1
        self.prf = int(reader.header.values["acquisition.laserPulseRate"])
        self.num_samples = int(reader.header.values["DataLocusCount"])
        self.num_shots = int(reader.header.values["EndShots"])

        # voltage range of DAQ Card
        self.daq_card_range = reader.header.values["acquisition.alazar.channelA.inputRange"] or '2V'  # default to 2V
        # split the value into a numerical value and unit and equalize to volts
        m = re.match(r'(?P<value>([\d\.e\-]+))(?P<unit>([MV|mV|mv|V|v]+)$)', self.daq_card_range)
        if m.group('unit').lower() == 'mv':
            # convert
            self.daq_card_range = int(m.group('value')) * 1e-3
        else:
            self.daq_card_range = int(m.group('value'))

        self.header_size_bytes = int(reader.header.values["HeaderSize_Bytes"])
        self.encoding = reader.header.values["DataEncoding"]
        self.shots_size = math.floor((reader.header.file_size - self.header_size_bytes) /
                                     (self.num_samples * self.get_data_type_byte_size(self.encoding)))

        self.position_first_sample = float(reader.header.values["PositionOfFirstSample_m"])
        self.sp_rng = [1, self.num_samples]  # ROI range of sample points
        self.dist_cf = 1
        self.zero_point = 0

        self.time_unit = 'time'
        self.time_rng = None
        self.shot_rng = [0, self.num_shots - 1]
        self.utc_offset = None  # ROI range of shots

        self.freq_rng = [20, math.floor(self.prf / 2)]  # analysis frequency bands
        self.time_rng_view = [20, math.floor(self.prf / 2)]  # frequency bands to view
        self.output_path = None  # output path

        self.num_fft = 2048

        if "Acquisition.Optics.PulseRepetitionFrequency_Hz" in reader.header.values:
            self.prf = int(reader.header.values["Acquisition.Optics.PulseRepetitionFrequency_Hz"])
        else:
            self.prf = int(reader.header.values["acquisition.laserPulseRate"])
        self.fft_bin_size = float(self.prf) / self.num_fft
        self.rolling_step = self.num_fft / 2  # 50% overlap
        self.rolling_step_percent = (self.rolling_step / self.num_fft) * 100
        self.rolling_sec = self.rolling_step / self.prf
        self.frame_length = self.num_fft

        self.num_frames = \
            int(math.floor((numpy.diff(self.shot_rng, n=1, axis=0)
                            + 1 - self.frame_length) / self.rolling_step) + 1)

        self.psd_stacking_factor = 1
        self.nf_adjustment = False

        self.display_mode = 3
        self.dpsd_c_rng = [0, 0]
        self.dsnr_y_rng = [0, 0]
        self.dsf_c_rng = [0, 0]

        self.b_animation = True
        self.b_save_animation = True
        self.b_save_frame = False
        self.b_save_sf = True
        self.b_save_result = False
        self.b_save_psd = False
        self.psd_file = ''

        self.processor = 'N-CPU'  # 'CPU', 'GPU', 'N-CPU'
        self.sf = []    # sound field data
        self.freq_vector = None
        self.dist_vector = [0, 0]  # get_dist_vector()
        self.time_vector = []

        # private properties
        self.fig = []
        self.noise_floor = None
        self.fft = None
        self.psd = None
        self.apsd = None
        self.snr = None
        self.psd_plot = []  # spectral animation
        self.psd_image_handle = []
        self.snr_plot = []  # snr animation
        self.sf_image_handle = []
        self.frame_shot_rng = None
        self.bin_rng = None
        self.v_bin_rng = None

        self.freq_vector = self.prf / 2 * numpy.linspace(0, 1, self.num_fft / 2 + 1)

        def my_func(x):
            v1 = round(self.freq_rng[0] / self.fft_bin_size) + 1
            v2 = round(self.freq_rng[1] / self.fft_bin_size) + 1
            return [v1, v2]

        self.bin_rng = map(my_func, range(1, len(self.freq_rng)))
        self.bin_rng = self.bin_rng[0]

        self.v_bin_rng = [round(self.time_rng_view[0] / self.fft_bin_size) + 1,
                          round(self.time_rng_view[1] / self.fft_bin_size) + 1]

        self.num_samples = int(numpy.diff(self.sp_rng, n=1, axis=0) + 1)

        self.num_shots = numpy.diff(self.shot_rng) + 2

        self.time_vector = self.get_time_vector()  # TODO: Fix this method but only used for display

        self.time_vector = numpy.empty(545)
        self.time_vector.fill(735670)

        # TODO: remove hard-coded values here.  This is only used for display
        self.dist_vector = [.9200, 3.4872, 4.0544, 4.6216, 5.1888,
                            5.7560, 6.3232, 6.8904, 7.4576, 8.0248,
                            8.5920, 9.1592, 9.7264, 10.2935, 10.8607,
                            11.4279, 11.9951, 12.5623, 13.1295, 13.6967,
                            14.2639, 14.8311, 15.3983, 15.9655, 16.5327]

    @staticmethod
    def print_output(title='', value=''):
        print("{0}\n\t{1}".format(title, value))

    @staticmethod
    def get_data_type_byte_size(encoding):
        """
        Get the machine dependent size of the data type used
        in the FDS data file
        :param encoding: The data type to find the size for
        :return: The number of bytes for the data type
        """
        if encoding in ['float32' 'real32' 'single']:
            data_byte = numpy.dtype(int).itemsize

        elif encoding in ['real64' 'double']:
            data_byte = numpy.dtype(float).itemsize

        elif encoding in ['uint16']:
            data_byte = numpy.dtype(numpy.uint16).itemsize

        else:
            data_byte = 4  # default

        return data_byte

    def get_time_vector(self):
        """
        Return a vector of time periods to divide the data
        :return:
        """
        sec_rng = numpy.subtract(self.shot_rng, 1) / self.prf
        step_s = self.rolling_sec * self.psd_stacking_factor
        vector = range(sec_rng[0], step_s, sec_rng[1])

        first_shot = self.reader.header.values["TimeOfFirstSample"]
        time_no = self.time_string_to_sec(first_shot, 1)

        # time_no = datenum(ref_time.Year,ref_time.Month,ref_time.Day,0,0,ref_seconds);
        vector = numpy.multiply(vector, (1 / (3600 * 24))) + time_no

        return vector

    def time_string_to_sec(self, time_string, format=1):
        """timestr2sec

        This method is used to convert time string to seconds.

        Usage:
           [struct seconds] = BaseTools.timestr2sec(time_string)

           * time_string can be a string or a column cell array {time1;time2;...}

        Input:
           format - i.e. 1 - 'HH:MM:SS.FFFFFFF'
                             'HH.MM.SS.FFFFFFF'
                         2 - 'HH:MM:SS.FFF'
                             'HH.MM.SS.FFF'
                         3 - 'yyyy-mm-ddTHH:MM:SS.FFFFFFF'
                             'yyyy:mm:ddTHH:MM:SS.FFFFFFF'
                             'yyyy.mm.dd.HH.MM.SS.FFFFFFF'
                         4 - 'HHMMSS.FFFFFFF'
                         5 - 'yyyymmdd'
                         6 - 'mmddyyyy'

        """
        if format == 1:
            # 'HH:MM:SS.FFFFFFF', 'HH.MM.SS.FFFFFFF'
            index = time_string.find(':')
            substring = time_string[index + 1:-2]

            # use regex to split the time
            segments = re.split(r'\s+|[:.]\s*', substring)

            hours = int(segments[0])
            minutes = int(segments[1])
            millisecs = float(segments[2]) / 1000000

            seconds = float((hours * 60 * 60) + (minutes * 60)) + millisecs

        return seconds

    def process_chunk(self, iteration, chunk):
        """
        Perform fft analysis on a data chunk
        :return:
        """
        # loop through all frames and run fft on each frame
        frame = iteration - 1

        # calculate the current shot range in this frame
        t = self.shot_rng[0] + (frame - 1) * self.rolling_step
        self.frame_shot_rng = [t + 1, t + self.frame_length]

        # read data
        if frame == 1:
            rows = numpy.shape(chunk)[0]
            buffer_shot_rng = [self.frame_shot_rng[0], self.frame_shot_rng[0] + rows - 1]  # TODO remove hardcode

        # generate Raw PSD
        fft = numpy.fft.fft(chunk, self.num_fft, axis=0)

        psd = abs(fft[0:self.num_fft / 2 + 1, :]) ** 2 / self.frame_length

        psd[1:-1] = 2 * psd[1:-1]  # ignore DC (0Hz) and Nyquist

        # stacking psd
        if self.psd_stacking_factor == 1:
            apsd = psd

        # estimate noise floor level
        if frame % self.psd_stacking_factor == 0 or frame == self.num_frames:
            self.noise_floor = numpy.median(apsd[(-self.num_fft / 4):-2,:], axis=0)

            self.noise_floor[self.noise_floor <= 0] = 1e-6

            # calculate SNR
            for i in xrange(0, len(self.freq_rng) / 2):

                # make an array of logically false values
                b_mask = numpy.zeros(apsd.shape).astype(int)

                # bin_rng is in the form [x y] where x is the start and y is the end
                # subtract one because python is zero indexed, unlike MATLAB
                bin_rng_start = self.bin_rng[i] - 1
                # add one because python's ending value is excluded, unlike MATLAB
                bin_rng_end = self.bin_rng[i + 1] + 1

                apsd_compare = apsd[bin_rng_start: bin_rng_end]
                replicated = numpy.tile(self.noise_floor, [int(numpy.diff(self.bin_rng)) + 1, 1])

                b_mask[bin_rng_start: bin_rng_end] = (apsd_compare > replicated)

                signal_est = sum(apsd[bin_rng_start: bin_rng_end] *
                                 b_mask[bin_rng_start: bin_rng_end], 0)

                not_b_mask = numpy.logical_not(b_mask[bin_rng_start: bin_rng_end])
                noise_est = sum(apsd_compare * not_b_mask, 0)

                with numpy.errstate(divide='ignore'):
                    self.snr = numpy.divide(signal_est, noise_est)

                self.snr[signal_est == 0] = 0
                self.snr[noise_est == 0] = 0

                # reset and insert the value at the beginning of the array
                self.sf = []
                self.sf = numpy.insert(self.sf, 0, self.snr, 0)

            # clear noise floor vector
            self.noise_floor = []

        return self.sf