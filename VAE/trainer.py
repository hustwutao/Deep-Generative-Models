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
        self.ckpt = tf.train.Checkpoint(enc=self.model.enc, dec=self.model.dec, optim=self.model.optim, epoch=self.model.global_epoch)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_dir, checkpoint_name='ckpt', max_to_keep=2)

        # tensorboard
        self.tensorboard_dir = make_dirs(os.path.join(self.config.result_path, self.config.tensorboard_path))
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)


    @tf.function    
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            batch_loss = self.model.loss(x_batch)

        t_vars = self.model.enc.trainable_variables + self.model.dec.trainable_variables
        enc_grads = tape.gradient(batch_loss, t_vars)

        self.model.optim.apply_gradients(zip(enc_grads, t_vars))
        self.model.global_step.assign_add(1)

        return batch_loss

    def train(self):

        # before training, loading checkpoint if exists
        self.load_model()
        start_epoch = self.model.global_epoch.numpy()

        for epoch in tqdm(range(start_epoch, self.config.num_epochs)):
            start = time.time()
            tot_loss = 0
            for step, (x_batch, _) in enumerate(self.trainloader):
                batch_loss = self.train_step(x_batch)
                tot_loss += batch_loss

            # write to tensorboard every epoch
            with self.summary_writer.as_default():
                tf.summary.scalar('tot_loss', tot_loss, epoch)

            print('epoch:{}, time:{:.2f}, tot_loss:{:.2f}'.format(epoch, time.time()-start, tot_loss.numpy()))
            self.model.global_epoch.assign_add(1)

            self.experiments.image_generation()
            self.save_model(epoch)

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
