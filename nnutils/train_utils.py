"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags

from ..utils.tf_visualizer import Visualizer as TfVisualizer

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')

flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('optim_bs', 1, 'Perform parameter update every optim_bs iterations')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')

flags.DEFINE_bool('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as momentum')

flags.DEFINE_integer('num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
flags.DEFINE_string('logging_dir', osp.join(cache_path, 'logs'),
                    'Root directory for log files')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 10000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 2, 'save model every k epochs')
flags.DEFINE_integer('lr_step_epoch_freq', 10, 'Reduce LR by factor of 10 every k ephochs')

## Flags for visualization
flags.DEFINE_integer('display_freq', 100, 'visuals logging frequency')
flags.DEFINE_integer('min_display_iter', 400, 'Skip plotting for initial iterations')
flags.DEFINE_boolean('display_visuals', True, 'whether to display images')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', True, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_integer('display_id', 1, 'Display Id')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_integer('display_port', 8097, 'Display port')
flags.DEFINE_integer('display_single_pane_ncols', 0, 'if positive, display all images in a single visdom web panel with certain number of images per row.')

#--------- training class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        torch.cuda.set_device(opts.gpu_id)

        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        self.log_dir = os.path.join(opts.logging_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        self.sc_dict = {}
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        print('Loading model')
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        return

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def get_current_visuals(self):
        return {}

    def get_current_scalars(self):
        return self.sc_dict

    def register_scalars(self, sc_dict, beta=0.99):
        '''
        Keeps a running smoothed average of some scalars.
        '''
        for k in sc_dict:
            if k not in self.sc_dict:
                self.sc_dict[k] = sc_dict[k]
            else:
                self.sc_dict[k] = beta*self.sc_dict[k] + (1-beta)*sc_dict[k]
        

    def get_current_points(self):
        return {}

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        if opts.use_sgd:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=opts.learning_rate, momentum=opts.beta1)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opts.learning_rate, betas=(opts.beta1, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opts.lr_step_epoch_freq, gamma=0.1)

    def train(self):
        opts = self.opts
        self.visualizer = TfVisualizer(opts)

        visualizer = self.visualizer
        total_steps = 0
        optim_steps = 0
        dataset_size = len(self.dataloader)

        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            self.scheduler.step()
            epoch_iter = 0
            self.curr_epoch = epoch
            for i, batch in enumerate(self.dataloader):
                t_init = time.time()
                self.set_input(batch)
                t_batch = time.time()

                if not self.invalid_batch:
                    optim_steps += 1
                    if optim_steps % opts.optim_bs == 0:
                        self.optimizer.zero_grad()

                    self.forward()
                    t_forw = time.time()
                    self.total_loss.backward()
                    t_backw = time.time()
                    # pdb.set_trace()
                    if optim_steps % opts.optim_bs == 0:
                        self.optimizer.step()

                    t_opt = time.time()

                    # print('t_batch: {}'.format(t_batch - t_init))
                    # print('t_forw: {}'.format(t_forw - t_batch))
                    # print('t_backw: {}'.format(t_backw - t_forw))
                    # print('t_opt: {}'.format(t_opt - t_backw))
                    # print('')

                total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    iter_end_time = time.time()
                    visualizer.log_images(self.get_current_visuals(), epoch*dataset_size + epoch_iter)

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    visualizer.print_current_scalars(epoch, epoch_iter, scalars)
                    if opts.plot_scalars  and (total_steps >= opts.min_display_iter):
                        visualizer.log_scalars(scalars, epoch*dataset_size + epoch_iter)
                        
                if total_steps % opts.save_latest_freq == 0:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return

            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)
