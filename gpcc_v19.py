import os, time
import numpy as np
import subprocess
from tqdm import tqdm
rootdir = os.path.split(__file__)[0]
if rootdir == '': rootdir = '.'


def get_points_number(filedir):
    plyfile = open(filedir)
    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])

    return number


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number

def gpcc_encode_v19(filedir, bin_dir, show=False):
    """Compress point cloud losslessly using MPEG G-PCCv14.
    You can download and install TMC13 from https://github.com/MPEGGroup/mpeg-pcc-tmc13
    """
    subp = subprocess.Popen(rootdir + '/tmc3v19' +
                            ' --mode=0' +
                            ' --positionQuantizationScale=1' +
                            ' --trisoupNodeSizeLog2=0' +
                            ' --neighbourAvailBoundaryLog2=8' +
                            ' --intra_pred_max_node_size_log2=6' +
                            ' --inferredDirectCodingMode=0' +
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath=' + filedir +
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return


def gpcc_decode_v19(bin_dir, rec_dir, show=False):
    subp = subprocess.Popen(rootdir + '/tmc3v19' +
                            ' --mode=1' +
                            ' --compressedStreamPath=' + bin_dir +
                            ' --reconstructedDataPath=' + rec_dir +
                            ' --outputBinaryPly=0'
                            ,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return
