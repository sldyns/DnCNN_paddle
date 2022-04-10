import cv2
import os
import argparse
import glob
import numpy as np
import paddle
import paddle.nn as nn
from models import DnCNN
from utils import *
from paddle.distribution import Normal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='BSD68', help='test on BSD68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)

    model.set_state_dict(paddle.load(os.path.join(opt.log_dir, 'net.pdparams')))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()

    psnr_test_all = []
    # test 10 times
    for i in range(10):
        print("Test for time: "+ str(i))

        # process data
        psnr_test = 0
        for f in files_source:
            # image
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:,:,0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)
            ISource = paddle.Tensor(Img)
            # noise

            normal = Normal([0.], [opt.test_noiseL / 255.])
            noise = normal.sample(ISource.shape)
            noise = paddle.squeeze(noise, axis=-1)

            # noisy image
            INoisy = ISource + noise

            with paddle.no_grad(): # this can save much memory
                Out = paddle.clip(model(INoisy), 0., 1.)
            ## if you are using older version of Pypaddle, paddle.no_grad() may not be supported
            # ISource, INoisy = Variable(ISource,volatile=True), Variable(INoisy,volatile=True)
            # Out = paddle.clip(INoisy-model(INoisy), 0., 1.)
            psnr = batch_PSNR(Out, ISource, 1.)
            psnr_test += psnr
            print("%s PSNR %f" % (f, psnr))
        psnr_test /= len(files_source)
        psnr_test_all.append(psnr_test)
        print("\nPSNR on test data %f" % psnr_test)

    print("\n10 times test on test data, Averate PSNR: {0}, Variance: {1}".format(np.mean(psnr_test_all), np.var(psnr_test_all)))

if __name__ == "__main__":
    main()
