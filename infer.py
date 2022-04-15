import os
from PIL import Image

import glob

from paddle.distribution import Normal


import cv2
import argparse
from utils import *
from models import DnCNN
import paddle

parser = argparse.ArgumentParser(description="DnCNN_Inference")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--log_dir", type=str, default="logs", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="data/BSD68/", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():

    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.save_path + '/denoised/', exist_ok=True)
    os.makedirs(opt.save_path + '/noised/', exist_ok=True)
    os.makedirs(opt.save_path + '/couple/', exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)

    model.set_state_dict(paddle.load(os.path.join(opt.log_dir, 'net.pdparams')))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
    files_source.sort()

    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = Img[:, :, 0]
        
        ISource = np.float32(Img)/255.
        ISource = np.expand_dims(ISource, 0)
        ISource = np.expand_dims(ISource, 1)
        ISource = paddle.Tensor(ISource)
        # noise

        normal = Normal([0.], [opt.test_noiseL / 255.])
        noise = normal.sample(ISource.shape)
        noise = paddle.squeeze(noise, axis=-1)

        # noisy image
        INoisy = ISource + noise

        with paddle.no_grad():  # this can save much memory
            Out = paddle.clip(model(INoisy), 0., 1.)

        psnr = batch_PSNR(Out, ISource, 1.)

        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))

        if opt.use_GPU:
            save_noised = np.uint8(255 * paddle.clip(INoisy, 0., 1.).detach().cpu().numpy().squeeze())  # back to cpu
            save_out = np.uint8(255 * Out.detach().cpu().numpy().squeeze())  # back to cpu
        else:
            save_noised = np.uint8(255 * paddle.clip(INoisy, 0., 1.).numpy().squeeze())  # back to cpu
            save_out = np.uint8(255 * Out.data.numpy().squeeze())

        save_couple = np.hstack([Img, np.zeros([save_out.shape[0], 16] ,dtype=np.uint8), save_noised,np.zeros([save_out.shape[0], 16] ,dtype=np.uint8), save_out])

        cv2.imwrite(os.path.join(opt.save_path, opt.data_path.split('/')[-1], 'noised/', f.split('/')[-1]),save_noised)
        cv2.imwrite(os.path.join(opt.save_path, opt.data_path.split('/')[-1], 'denoised/', f.split('/')[-1]), save_out)
        cv2.imwrite(os.path.join(opt.save_path, opt.data_path.split('/')[-1], 'couple/', f.split('/')[-1]), save_couple)


    psnr_test /= len(files_source)

    print("\nPSNR on test data %f" % psnr_test)



if __name__ == "__main__":
    main()
