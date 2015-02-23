__author__ = 'o1806'

from FdsHeader import FdsHeader


class ReadFDS(object):
    """
    Read an FDS file in chunks so that
    we will be able to process large files (>2GB)
    """
    mat = None

    def __init__(self, file_in, file_out):
        self.file_in = file_in
        self.file_out = file_out
        self.mat = None
        self.header = None

    def read_header(self):
        """
        Get basic information from the header so we know
        where the data starts and information about the data
        :return:
        """
        self.header = FdsHeader(debug=False)
        self.header.process(self.file_in)

    def read_chunks(self, data_start_loc, chunk_size):
        with open(self.file_in, 'rb') as in_file:
            in_file.seek(data_start_loc)

            for chunk in iter(lambda: in_file.read(chunk_size), ''):
                yield chunk
