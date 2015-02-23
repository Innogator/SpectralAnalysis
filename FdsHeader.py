__author__ = 'o1806'

"""
    [HeaderStruct ASCIIBlocks dataStartLoc] = GetFDSHeader(FileName) takes a
    path to an FDS file and returns a struct containing the Key/value pairs
    in the FDS Header, a cell array of unparsed ASCII blocks representing the
    header sections used for the FDEL properties file, acquisition block, or
    padding block, and the integer location in bytes of the start of the data
    section of the FDS file.
"""

import sys
import os
import numpy as np


class FdsHeader(object):

    DEBUG = None

    def __init__(self, debug=False):
        self.DEBUG = debug

        # dictionary for all values in the header
        self.values = {}

        self.fds_version = None
        self.data_start_loc = 0
        self.header_section_sizes_bytes = None
        self.header_section_labels = None
        self.num_rows = None
        self.data_encoding = None
        self.num_ascii_blocks = None
        self.header_section_ends = None
        self.ascii_blocks = []

        self.file_size = None

    @staticmethod
    def get_header_value(string):
        values = string.split('=')
        return values[1].strip()

    @staticmethod
    def strip_line(string):
        string = string.strip('[')
        string = string.strip(']')
        string = string.strip()

        return string

    def process(self, f):
        self.file_size = os.stat(f).st_size

        with open(f, "rb") as infile:

            if self.DEBUG:
                print("Begin processing: ")

            # read FDS version
            line = infile.readline()
            self.fds_version = self.get_header_value(line)
            if self.DEBUG:
                print("fds_version: " + self.fds_version)

            # get the data start location
            line = infile.readline()
            self.data_start_loc = int(self.get_header_value(line))
            if self.DEBUG:
                print("data_start_loc: " + str(self.data_start_loc))

            # get header_section_sizes_bytes
            line = infile.readline()
            line = self.get_header_value(line)
            line = self.strip_line(line)
            self.header_section_sizes_bytes = line.split()
            # convert the string values to ints
            self.header_section_sizes_bytes = map(int, self.header_section_sizes_bytes)
            if self.DEBUG:
                print("header_section_bytes: ")
                print(self.header_section_sizes_bytes)

            # get header_section labels
            line = infile.readline()
            line = self.get_header_value(line)
            line = self.strip_line(line)
            self.header_section_labels = line.split()
            if self.DEBUG:
                print("header_section_labels: ")
                print(self.header_section_labels)

            # get num_rows for the matrix
            line = infile.readline()
            self.num_rows = int(self.get_header_value(line))
            if self.DEBUG:
                print("num_rows: " + str(self.num_rows))

            # get data_encoding types
            line = infile.readline()
            self.data_encoding = self.get_header_value(line)
            if self.DEBUG:
                print("data_encoding: " + self.data_encoding)

            # make sure all values are digits
            if not all(type(item) is int for item in self.header_section_sizes_bytes):
                raise Exception("readFDSHeader:InvalidHeader\nHeaderSectionSizes_Bytes contains non-digits")
            elif len(self.header_section_sizes_bytes) < 2:
                raise Exception("readFDSHeader:InvalidHeader\nHeaderSectionSizes_Bytes doesn't contain enough values.")
            elif len(self.header_section_sizes_bytes) == 2:
                print("No ASCII block size specified.  Assuming no ASCII block.")
                self.header_section_sizes_bytes.append(0)
            else:
                # end of the header section / beginning of te first ASCII block
                self.num_ascii_blocks = len(self.header_section_sizes_bytes) - 2
                if self.DEBUG:
                    print("num_ascii_blocks: " + str(self.num_ascii_blocks))

                # vector of byte positions of the ends of the various header sections
                self.header_section_ends = np.cumsum(self.header_section_sizes_bytes)
                if self.DEBUG:
                    print("header_section_ends: " + str(self.header_section_ends))

            bytes_to_read = int(self.header_section_ends[-1])
            # parse tags and values
            # reset the file position to zero to read all values
            infile.seek(0)
            header_section = infile.readlines(self.header_section_sizes_bytes[-1])
            del header_section[-1]
            if self.DEBUG:
                print("header_section: ")
                for line in header_section:
                    print(line)

            try:
                for line in header_section:
                    split = line.split('=')
                    key = split[0].strip()

                    if len(split) > 1:
                        value = split[1].strip()
                    else:
                        value = None

                    if self.DEBUG:
                        print("{0} = {1}".format(key, value))

                    self.values[key] = value

            except:
                e = sys.exc_info()[0]
                print("Error: " + str(e))

