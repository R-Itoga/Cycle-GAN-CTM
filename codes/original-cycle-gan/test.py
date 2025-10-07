import os
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from cycle_gan_data import cycle_gan_data_source, cycle_gan_data_iterator
import models
from helpers import MonitorImageWithName


def test(args):
    # Settings
    b = args.batch_size 
    c, h, w = 3, 256, 256
    beta1 = 0.5
    beta2 = 0.999
    lambda_recon = args.lambda_recon
    lambda_idt = args.lambda_idt
    base_lr = args.learning_rate
    init_method = args.init_method

    # Context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Inputs
    x_real_test = nn.Variable([b, c, h, w], need_grad=False)
    y_real_test = nn.Variable([b, c, h, w], need_grad=False)

    # Models for test
    model_path = args.model_load_path
    
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'params.h5')

    logger.info(f"Loading test model from: {model_path}")
    nn.load_parameters(model_path)
    y_fake_test = models.g(
        x_real_test, unpool=args.unpool, init_method=init_method)
    x_fake_test = models.f(
        y_real_test, unpool=args.unpool, init_method=init_method)
    y_fake_test.persistent, x_fake_test.persistent = True, True
    # Reconstruct
    x_recon_test = models.f(
        y_fake_test, unpool=args.unpool, init_method=init_method)
    y_recon_test = models.g(
        x_fake_test, unpool=args.unpool, init_method=init_method)

    # Datasets
    rng = np.random.RandomState(313)
    ds_test_B = cycle_gan_data_source(
        args.dataset, train=False, domain="B", shuffle=False, rng=rng)
    ds_test_A = cycle_gan_data_source(
        args.dataset, train=False, domain="A", shuffle=False, rng=rng)
    di_test_B = cycle_gan_data_iterator(ds_test_B, args.batch_size)
    di_test_A = cycle_gan_data_iterator(ds_test_A, args.batch_size)

    # Monitors
    monitor = Monitor(args.monitor_path)

    def make_monitor_image(name):
        # return MonitorImageWithName(name, monitor, interval=1,
        #                         normalize_method=lambda x: (x + 1.0) * 127.5)
        return MonitorImageWithName(name, monitor, interval=1,
                                    normalize_method=lambda x: x + 1.0)
    monitor_test_gx = make_monitor_image('fake_images_test_A')
    monitor_test_fy = make_monitor_image('fake_images_test_B')
    monitor_test_x_recon = make_monitor_image('fake_images_recon_test_B')
    monitor_test_y_recon = make_monitor_image('fake_images_recon_test_A')

    # Validation for B
    logger.info("Validation for B")
   
    for i in range((di_test_A.size + args.batch_size - 1) // args.batch_size):
        y_data, _ = di_test_A.next()
        y_real_test.d = y_data
        y_recon_test.forward(clear_buffer=True)
        
        
        for j in range(y_data.shape[0]):
            file_index = i * args.batch_size + j
            
            if file_index >= di_test_A.size:
                continue
            
            name = ds_test_A.filename_list[file_index]
            logger.info("generating a fake of {}".format(name))
            
            
            fake_b_image = np.expand_dims(x_fake_test.d[j], axis=0)
            recon_a_image = np.expand_dims(y_recon_test.d[j], axis=0)
            
            monitor_test_fy.add(name, fake_b_image)
            monitor_test_y_recon.add(name, recon_a_image)

    # Validation for A
    logger.info("Validation for A")
    
    for i in range((di_test_B.size + args.batch_size - 1) // args.batch_size):
        x_data, _ = di_test_B.next()
        x_real_test.d = x_data
        x_recon_test.forward(clear_buffer=True)

        
        for j in range(x_data.shape[0]):
            file_index = i * args.batch_size + j
            
            if file_index >= di_test_B.size:
                continue
            
            name = ds_test_B.filename_list[file_index]
            logger.info("generating a fake of {}".format(name))

           
            fake_a_image = np.expand_dims(y_fake_test.d[j], axis=0)
            recon_b_image = np.expand_dims(x_recon_test.d[j], axis=0)

            monitor_test_gx.add(name, fake_a_image)
            monitor_test_x_recon.add(name, recon_b_image)


def main():
    args = get_args()
    save_args(args)
    test(args)


if __name__ == '__main__':
    main()
