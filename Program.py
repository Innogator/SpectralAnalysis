__author__ = 'o1806'


# local imports
from ReadFDS import ReadFDS
from SpectralAnalysis import SpectralAnalysis
from FilePrompt import get_file_path

# module imports
import wx
import time
import numpy as np
from matplotlib import pyplot as plt


class bcolors:
    """
    Colors used to print to terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():

    animate = False

    # get the file to open
    file_in = get_file_path()
    file_out = file_in + '_original.txt'
    #file_processed = '../' + file_main + '_processed.txt'

    start = time.time()

    # set console color to green
    print(bcolors.OKGREEN)

    # define the file reader
    reader = ReadFDS(file_in, file_out)
    reader.read_header()

    # define the analyzer
    analyzer = SpectralAnalysis(reader)

    # initializers
    num_frames = analyzer.num_frames
    num_fft = analyzer.num_fft
    num_fft_half = num_fft / 2
    processed = np.zeros((num_frames, reader.header.num_rows))
    mat = np.zeros((num_fft, reader.header.num_rows))

    # TODO: Calculate bytes_to_read dynamically
    # TODO: i.e. frame_size * data_type_size
    bytes_to_read = num_fft * reader.header.num_rows
    iteration = 0

    if animate:
        plt.figure()
        plt.imshow(processed, aspect='auto')

    for chunk in reader.read_chunks(reader.header.data_start_loc, bytes_to_read):
        print("{0} of {1}".format(iteration, num_frames + 1))

        array = np.fromstring(chunk, dtype=np.uint16)

        # convert bad values to zero, this is done in MATLAB as well
        array = np.nan_to_num(array)
        array = (np.reshape(array, (-1, reader.header.num_rows)))

        # roll the values down the matrix to be replaced by
        # the next rolling step
        mat = np.roll(mat, num_fft_half, axis=0)

        if array.shape[0] != num_fft_half:
            break
            mat[num_fft_half:] = 0
            print("shape {0}".format(array.shape))
            mat[num_fft_half:num_fft_half + array.shape[0]] = array
        else:
            mat[num_fft_half:] = array
            iteration += 1

        #print("mat size {0}".format(mat.shape))

        # write this chunk to the "_original" file
        # TODO
        # it takes 2 iterations to have a set because the
        # rolling step is half the size of an FFT process
        if iteration > 1:
            # insert the new values at the beginning of the results array
            new_processed = analyzer.process_chunk(iteration, mat)

            # roll the data down a row to insert new data
            processed = np.roll(processed, 1, axis=0)
            processed[0] = new_processed

            #processed = np.insert(processed, 0, new_processed, 0)

            if animate and (iteration % 5 == 0 or iteration, num_frames + 1):
                #image.set_data(processed)
                image = plt.imshow(processed, aspect='auto')
                plt.draw()
                plt.pause(0.01)
                plt.clf()

    end = time.time()

    print(bcolors.OKBLUE)
    print ("Total Elapsed Time: {0}".format(end - start))

    print(bcolors.ENDC)

    plt.imshow(processed, aspect='auto')
    plt.colorbar()
    plt.show()

    #filename = '../part2.5B_2014.07.08.17.29.59_processed.txt'
    #fd = open(filename,'wb')
    #np.savetxt(filename, processed, delimiter=" ", fmt="%f")
    #fd.close()


if __name__ == '__main__':
    main()