import os
import argparse

import paddle

from models import DnCNN

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--savedir", type=str, default=".", help='path of model for export')

opt = parser.parse_args()


def main(opt):

    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)

    if opt.logdir:
        model.set_state_dict(paddle.load(os.path.join(opt.logdir, 'net.pdparams')))
        print('Loaded trained params of model successfully.')

    shape = [-1, 1, 256, 256]

    new_model = model

    new_model.eval()
    new_net = paddle.jit.to_static(
        new_model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(opt.save_dir, 'model')
    paddle.jit.save(new_net, save_path)


    print(f'Model is saved in {opt.save_dir}.')


if __name__ == '__main__':
    main(opt)