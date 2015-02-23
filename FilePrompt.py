__author__ = 'o1806'

import Tkinter
import tkFileDialog


def get_file_path():

    root = Tkinter.Tk()
    root.withdraw()

    file_path = tkFileDialog.askopenfilename()
    return file_path
