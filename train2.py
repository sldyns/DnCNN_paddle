import os
import argparse
import numpy as np
import paddle
import paddle.optimizer as optim
import paddle.vision
from paddle.io import DataLoader
from models import DnCNN
from dataset import prepare_data, DnCNN_Dataset
from utils import *
from paddle.distribution import Normal
from visualdl import LogWriter
import paddle.distributed as dist

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=10, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default="data/train", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="data/BSD68", help="path of val dataset")
parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=15, help='noise level')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
opt = parser.parse_args()
print(opt)


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = DnCNN_Dataset(train=True)
    dataset_val = DnCNN_Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)

    print("# of training samples: %d\n" % int(len(dataset_train)))
    dist.init_parallel_env()

    # Build model
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    print(model)

    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU

    model = paddle.DataParallel(model)

    # Optimizer
    scheduler = optim.lr.ExponentialDecay(learning_rate=opt.lr, gamma=0.9, verbose=True)
    optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=scheduler)
    # training
    with LogWriter(logdir=opt.log_dir) as writer:
        step = 0
        best_val = 0

        for epoch in range(opt.epochs):

            # train
            model.train()

            for i, img_train in enumerate(loader_train, 0):

                # training step

                normal = Normal([0.], [opt.noiseL / 255.])
                noise = normal.sample(img_train.shape)
                noise = paddle.squeeze(noise, axis=-1)

                noise.stop_gradient = False
                img_train.stop_gradient = False

                imgn_train = img_train + noise

                out_train = model(imgn_train)
                loss = criterion(out_train, img_train) / (img_train.shape[0] * 2.)

                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

                # results
                if step % 200 == 0:
                    psnr_train = batch_PSNR(out_train, img_train, 1.)

                    print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                          (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

                    # Log the scalar values

                    writer.add_scalar(tag='loss', value=loss.item(), step=step)
                    writer.add_scalar(tag='PSNR on training data', value=psnr_train, step=step)
                step += 1

            ## the end of each epoch
            scheduler.step()

            model.eval()
            # validate
            psnr_val = 0
            normal = Normal([0.], [opt.val_noiseL / 255.])

            for img_val in loader_val:
                noise = normal.sample(img_val.shape)
                noise = paddle.squeeze(noise, axis=-1)

                imgn_val = img_val + noise

                out_val = paddle.clip(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))

            writer.add_scalar(tag='PSNR on validation data', value=psnr_val, step=epoch)

            if psnr_val > best_val:
                paddle.save(model.state_dict(), os.path.join(opt.log_dir, 'net.pdparams'))
                best_val = psnr_val


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path=opt.data_dir, val_path=opt.val_dir, patch_size=40, stride=10, aug_times=2)
    main()
