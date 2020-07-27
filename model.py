from __future__ import division
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import time
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import namedtuple

from module import *
from utils import *
import utils

import cv2

class vae(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        #self.image_size = args.fine_size
        self.L1_lambda = args.L1_lambda
        
        self.alpha = args.alpha

        self.feature_extraction_network = feature_extraction_network
        self.prediction_network = prediction_network
        self.recognition_network = recognition_network
        self.reconstruction_network = reconstruction_network
        self.mse = mse_criterion
        self.checkpoint_dir = args.pj_dir + 'checkpoint'
        self.logs_dir = args.pj_dir + 'logs'
        self.sample_dir = args.pj_dir + 'sample'
        self.test_dir =  args.pj_dir + 'test'
        self.dataset_dir = args.pj_dir + 'data'


        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)


        self._build_model(args)
        
        self.saver = tf.train.Saver(max_to_keep=100)
        
        self._load_dataset(args)

    def _load_dataset(self, args):
        if len(os.listdir(self.dataset_dir)) == 0:
            gdd.download_file_from_google_drive(file_id='1F1l1c0D4unefRK9m6i2WqBO7dPvkbdKQ',
                                        dest_path=self.dataset_dir+'/data.zip',
                                        unzip=True)
        if args.phase == 'train':
            self.ds = pd.read_csv(self.dataset_dir+'/KMAC_new_RCWA_raw_dataset_200612_norm_filename_0.7_train.csv')
        else:
            self.ds = pd.read_csv(self.dataset_dir+'/KMAC_new_RCWA_raw_dataset_200612_norm_filename_0.7_test.csv')

    def _load_batch(self, dataset, idx):
        
        filename_list = dataset.iloc[:,0][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist()

        # input batch (2d binary image)
        input_batch = []
        for i in range(len(filename_list)):
            temp_img = cv2.imread(self.dataset_dir+'/64/'+filename_list[i], 0)
            temp_img = temp_img/255
            input_batch.append(list(temp_img))
        input_batch = np.expand_dims(input_batch, axis=3)

        target_batch = np.expand_dims(dataset.iloc[:,6:][idx * self.batch_size:(idx + 1) * self.batch_size].values.tolist(), 2) # [0.543, ... ] 226개

        return input_batch, target_batch, filename_list


    def _build_model(self, args):
        # ref : https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
        # log sigma ref : https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/


        self.geo_labeled = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input_l')
        #self.geo_unlabeled = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input_u')
        self.spectrum_target = tf.placeholder(tf.float32, [None, 202, 1], name='spectra_target')
        self.latent_vector = tf.placeholder(tf.float32, [None, args.latent_dims], name='latent_vector')

        # labeled data sequence
        feature_l = self.feature_extraction_network(self.geo_labeled, reuse=False)
        self.spectra_l_predicted = self.prediction_network(feature_l, reuse=False)
        mu, log_sigma = self.recognition_network(feature_l, self.spectrum_target, args.latent_dims, reuse=False)
        self.latent_variable_a = mu + tf.exp(log_sigma/2) * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        self.geo_reconstructed_l = self.reconstruction_network(self.spectrum_target, self.latent_variable_a, reuse=False)

        # reconstruction (inverse generation) model
        self.geo_reconstructed = self.reconstruction_network(self.spectrum_target, self.latent_vector, reuse=True)

        ## unlabeled data sequence (사용 안함)
        #feature_u = self.feature_extraction_network(self.geo_unlabeled, reuse=True)
        #spectra_u = self.prediction_network(feature_u, reuse=True)
        #mean_u, covariance_u = self.recognition_network(feature_u, spectra_u, reuse=True)
        #latent_variable_b = mean_u + covariance_u * tf.random_normal(tf.shape(mean_u), 0, 1, dtype=tf.float32)
        #geo_reconstructed_u = self.reconstruction_network(spectra_u, latent_variable_b, reuse=True)
        

        # Labeled data loss
        # KL div loss
        self.KL_div_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_sigma) - log_sigma - 1., 1)
        self.reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.geo_labeled,[self.batch_size,64*64]),
                                                                    logits=tf.reshape(self.geo_reconstructed_l,[self.batch_size,64*64]))
        self.loss_l = args.beta*tf.reduce_mean(self.KL_div_loss) + tf.reduce_mean(self.reconstruction_loss)


        ## Unlabeled data loss (검토 필요)
        #KL_div_u = 0.5 * tf.reduce_sum(tf.square(mean_u) + tf.square(covariance_u) - tf.log(1e-8 + tf.square(covariance_u)) - 1, 1)
        ##marginal_likelihood_u = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.geo_unlabeled,[self.batch_size,32*32]),
        ##                                                            logits=tf.reshape(geo_reconstructed_u,[self.batch_size,32*32]))
        #marginal_likelihood_u = tf.reduce_sum(self.geo_unlabeled * tf.log(1e-10+geo_reconstructed_u) + (1 - self.geo_unlabeled) * tf.log(1e-10+1 - geo_reconstructed_u), 1)
        #self.loss_u = tf.reduce_mean(KL_div_u) - tf.reduce_mean(marginal_likelihood_u)

        # Regression loss
        self.loss_r = self.alpha * self.mse(self.spectra_l_predicted, self.spectrum_target)

        # Total loss
        #self.total_loss = self.loss_l + self.loss_u + self.loss_r
        self.total_loss = self.loss_l + self.loss_r



        self.loss_summary = tf.summary.scalar("loss", self.total_loss)

        self.t_vars = tf.trainable_variables()
        print("trainable variables : ")
        print(self.t_vars)
        

    def train(self, args):
        
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, args.epoch_step, 0.96, staircase=False)

        self.optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1) \
            .minimize(self.total_loss, var_list=self.t_vars, global_step = global_step)
        #self.optim = tf.train.GradientDescentOptimizer(learning_rate) \
        #    .minimize(self.total_loss, var_list=self.t_vars, global_step = global_step)

        print("initialize")
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        
        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(self.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            
            batch_idxs = len(self.ds) // self.batch_size

            ds_1 = self.ds.sample(frac=1)
            
            for idx in range(0, batch_idxs):

                input_batch, target_batch, _ = self._load_batch(ds_1, idx)

                # Update network
                kl, marginal, la_v, geo_re, _, loss, loss_l,loss_r, c_lr, summary_str = self.sess.run([self.KL_div_loss,self.reconstruction_loss, self.latent_variable_a, self.geo_reconstructed_l, self.optim, self.total_loss, self.loss_l, self.loss_r, learning_rate, self.loss_summary],
                                                   feed_dict={self.geo_labeled: input_batch, self.spectrum_target: target_batch, self.lr: args.lr})

                self.writer.add_summary(summary_str, counter)

                counter += 1
                if idx%10==0:
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %4.4f loss_label: %4.4f loss_reg: %4.4f lr: %4.7f loss_kl: %4.7f loss_recon: %4.7f" % (
                        epoch, idx, batch_idxs, time.time() - start_time, loss,loss_l,loss_r, c_lr, np.mean(kl), np.mean(marginal))))

                if np.mod(counter, args.save_freq) == 20:
                    self.save(self.checkpoint_dir, counter)

            if epoch%1 == 0: # save sample image
                cv2.imwrite(self.sample_dir + '/epoch_'+str(epoch)+'_pred.bmp',(geo_re[0,:,:,0])*255)
                cv2.imwrite(self.sample_dir + '/epoch_'+str(epoch)+'_input.bmp',(input_batch[0,:,:,0])*255)

    def save(self, checkpoint_dir, step):
        model_name = "dnn.model"
        model_dir = "%s" % (self.dataset_dir)
        #checkpoint_dir = checkpoint_dir + '/' + model_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        checkpoint_dir+'/'+model_name,
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths
            print(ckpt_paths)
            ckpt_name = os.path.basename(ckpt_paths[-1])
            #temp_ckpt = 'dnn.model-80520'
            #ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def test(self, args):

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0

        batch_idxs = len(self.ds) // self.batch_size

        ds_1 = self.ds
        
        loss_list = []

        df_param_target_all = pd.DataFrame()
        df_param_pred_all = pd.DataFrame()

        for idx in range(0, batch_idxs):

            input_batch, target_batch, _ = self._load_batch(ds_1, idx)

            geo_pred, pred, loss, loss_r = self.sess.run([self.geo_reconstructed_l, self.spectra_l_predicted, self.total_loss, self.loss_r],
                                                feed_dict={self.geo_labeled: input_batch, self.spectrum_target: target_batch})


            loss_list.append(loss_r)

            counter += 1
            if idx%1==0:
                print(("Step: [%4d/%4d] time: %4.4f" % (
                    idx, batch_idxs, time.time() - start_time)))
                #df_param = pd.DataFrame(np.squeeze(input_batch), columns={'param1','param2','param3','param4','param5'}) 
                df_pred = pd.DataFrame(np.squeeze(pred))
                df_target = pd.DataFrame(np.squeeze(target_batch))
                #df_geo_pred =  np.squeeze(geo_pred)

                #df_param_pred = pd.concat([df_param, df_pred], axis=1, sort=False)
                #df_param_target = pd.concat([df_param, df_target], axis=1, sort=False)
                #df_param_param = pd.concat([df_param, df_geo_pred], axis=1, sort=False)
                
                df_param_target_all = pd.concat([df_param_target_all, df_target], axis=0, sort=False)
                df_param_pred_all = pd.concat([df_param_pred_all, df_pred], axis=0, sort=False)

            #df_param_target_all.to_csv(self.test_dir+'/result_test_target.csv', index=False)
            #df_param_pred_all.to_csv(self.test_dir+'/result_test_prediction.csv', index=False)

        print("mean regression loss : ")
        print(np.mean(loss_list)/args.alpha)
        print("total time")
        print(time.time() - start_time)


    def test_reconstruction(self, args):

        self.batch_size = 1

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 0

        batch_idxs = len(self.ds) // self.batch_size

        ds_1 = self.ds
        
        loss_list = []

        for idx in range(0, batch_idxs):

            input_batch, target_batch, filename_list = self._load_batch(ds_1, idx)

            for j in range(5):
                latent_vector = list(np.random.normal(0,3,5))
                print(latent_vector)
                latent_vector = np.expand_dims(latent_vector, 0)
                geo_recon = self.sess.run([self.geo_reconstructed], 
                                            feed_dict={self.latent_vector: latent_vector, self.spectrum_target: target_batch})


                print(self.test_dir+'/reconstruction/')
                geo_recon = np.squeeze(geo_recon)
                cv2.imwrite(self.test_dir+'/'+str(filename_list)+'_'+str(latent_vector)+'.bmp',(geo_recon+1)*128)
            