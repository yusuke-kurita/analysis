import os
import sys
import argparse

import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5

from misc import low_cut_filter
from yml import SpeakerYML

parser = argparse.ArgumentParser()

parser.add_argument('file', metavar='N', type=str)
parser.add_argument('-min', '--minf0', type=int, default=240)
parser.add_argument('-max', '--maxf0', type=int, default=700)
args = parser.parse_args()


def main():

    if args.file == 'con':
        file = 'converted'
    elif args.file == 'tar':
        file = 'target'
    else:
        raise ValueError("The file is incorrect")

    feat = FeatureExtractor(analyzer='world',
                            fs=22050,
                            fftl=1024,
                            shiftms=5,
                            minf0=args.minf0,
                            maxf0=args.maxf0)

    # constract Synthesizer class
    synthesizer = Synthesizer(fs=22050,
                              fftl=1024,
                              shiftms=5)

    # open list file
    with open('./list/' + file + '.list', 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join('./' + file + '/h5f/', f + '.h5')

            if (not os.path.exists(h5f)):
                wavf = os.path.join('./' + file + '/wav/', f + '.wav')
                fs, x = wavfile.read(wavf)
                x = np.array(x, dtype=np.float)
                x = low_cut_filter(x, fs, cutoff=70)

                print("Extract acoustic features: " + wavf)

                # analyze F0, spc, and ap
                f0, spc, ap = feat.analyze(x)
                mcep = feat.mcep(dim=34, alpha=0.544)
                npow = feat.npow()
                codeap = feat.codeap()

                # save features into a hdf5 file
                h5 = HDF5(h5f, mode='w')
                h5.save(f0, ext='f0')
                # h5.save(spc, ext='spc')
                # h5.save(ap, ext='ap')
                h5.save(mcep, ext='mcep')
                h5.save(npow, ext='npow')
                h5.save(codeap, ext='codeap')
                h5.close()

                # analysis/synthesis using F0, mcep, and ap
                wav = synthesizer.synthesis(f0,
                                            mcep,
                                            ap,
                                            alpha=0.544,
                                            )
                wav = np.clip(wav, -32768, 32767)
                anasynf = os.path.join('./' + file + '/anasyn/', f + '.wav')
                wavfile.write(anasynf, fs, np.array(wav, dtype=np.int16))
            else:
                print("Acoustic features already exist: " + h5f)


if __name__ == '__main__':
    main()
