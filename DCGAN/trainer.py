import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import data_loader, make_dirs

class Trainer(object):
    def __init__(self, config, model, experiments):
        self.config = config
        self.model = model
        self.experiments = experiments
        self.trainloader, _ = data_loader(config)

        # checkpoint
        self.checkpoint_dir = make_dirs(os.path.join(self.config.result_path, self.config.checkpoint_path))
        self.ckpt = tf.train.Checkpoint(gen=self.model.gen, g_optim=self.model.g_optim, dis=self.model.dis, d_optim=self.model.d_optim, epoch=self.model.global_epoch)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_dir, checkpoint_name='ckpt', max_to_keep=1)

        # tensorboard
        self.tensorboard_dir = make_dirs(os.path.join(self.config.result_path, self.config.tensorboard_path))
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        tf.summary.trace_on(graph=True, profiler=True)

    @tf.function    
    def train_step(self, x_batch):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            d_loss, g_loss = self.model.loss(x_batch)

        d_grads = d_tape.gradient(d_loss, self.model.dis.trainable_variables)
        self.model.d_optim.apply_gradients(zip(d_grads, self.model.dis.trainable_variables))

        g_grads = g_tape.gradient(g_loss, self.model.gen.trainable_variables)      
        self.model.g_optim.apply_gradients(zip(g_grads, self.model.gen.trainable_variables))

        self.model.global_step.assign_add(1)

        return d_loss, g_loss

    def train(self):

        # before training, loading checkpoint if exists
        self.load_model()
        start_epoch = self.model.global_epoch.numpy()

        for epoch in tqdm(range(start_epoch, self.config.num_epochs)):
            epoch_d_loss = 0
            epoch_g_loss = 0
            start = time.time()
            
            for step, (x_batch, _) in enumerate(self.trainloader):
                d_loss, g_loss = self.train_step(x_batch)
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss

            # save log file for tensorboard visualization every epoch
            with self.summary_writer.as_default():
                tf.summary.scalar('d_loss', epoch_d_loss, epoch)
                tf.summary.scalar('g_loss', epoch_g_loss, epoch)
                

            print('epoch:{}, time:{:.2f}, d_loss:{:.2f}, g_loss:{:.2f}'.format(epoch, time.time()-start, 
                                                                               epoch_d_loss.numpy(), epoch_g_loss.numpy()))
            self.model.global_epoch.assign_add(1)

            self.experiments.image_generation()
            self.save_model(epoch)

        with self.summary_writer.as_default():
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=self.tensorboard_dir)


    # save function that saves the checkpoint
    def save_model(self, epoch):
        print("Saving model...")
        self.ckpt_manager.save(checkpoint_number=epoch)
        print("Model saved")

    # load latest checkpoint
    def load_model(self):
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        else:
            print("Initializing from scratch.")
