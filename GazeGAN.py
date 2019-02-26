# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf
from Dataset import save_images
from ops import conv2d, lrelu, instance_norm, de_conv, fully_connect


class Gaze_GAN(object):

    # build model
    def __init__(self, dataset, config):


        self.dataset = dataset

        # input hyper
        self.output_size = config.image_size
        self.channel = dataset.channel
        self.batch_size = config.batch_size
        self.pos_number = config.pos_number
        self.pretrain_model_index = config.pretrain_model_index
        self.pretrain_model_dir = config.pretrain_model_dir

        # output hyper
        self.sample_dir = config.sample_dir
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        self.result_dir = config.result_dir
        self.batch_num = dataset.test_num / self.batch_size

        # model hyper
        self.lam_percep = config.lam_percep
        self.lam_recon = config.lam_recon
        self.loss_type = config.loss_type
        self.use_sp = config.use_sp
        self.log_vars = []

        # trainning hyper
        self.g_learning_rate = config.g_learning_rate
        self.d_learning_rate = config.d_learning_rate

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.lr_init = config.learning_rate_init
        self.start_step = config.start_step
        self.max_iters = config.max_iters

        # placeholder
        self.input_left_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input_right_labels = tf.placeholder(tf.float32, [self.batch_size, self.pos_number])
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.domain_label = tf.placeholder(tf.int32, [self.batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self.incomplete_img = self.input * (1 - self.mask)

        # self.local_input_left = tf.image.crop_and_resize(self.input, boxes=self.input_left_labels,
        #                         box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])
        # self.local_input_right = tf.image.crop_and_resize(self.input, boxes=self.input_right_labels,
        #                          box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])
        self.local_input_left = self.crop_and_resize(self.input, self.input_left_labels)
        self.local_input_right = self.crop_and_resize(self.input, self.input_right_labels)

        self.angle_invar_left_real = self.encode(self.local_input_left, reuse=False)
        self.angle_invar_right_real = self.encode(self.local_input_right, reuse=True)

        self.recon_img = self.generator(self.incomplete_img, self.mask, self.angle_invar_left_real,
                                        self.angle_invar_right_real, reuse=False)

        # self.local_recon_img_left = tf.image.crop_and_resize(self.recon_img, boxes=self.input_left_labels,
        #                          box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])
        # self.local_recon_img_right = tf.image.crop_and_resize(self.recon_img, boxes=self.input_right_labels,
        #                          box_ind=range(0, self.batch_size), crop_size=[self.output_size/2, self.output_size/2])

        self.local_recon_img_left = self.crop_and_resize(self.recon_img, self.input_left_labels)
        self.local_recon_img_right = self.crop_and_resize(self.recon_img, self.input_right_labels)

        self.angle_invar_left_recon = self.encode(self.local_recon_img_left, reuse=True)
        self.angle_invar_right_recon = self.encode(self.local_recon_img_right, reuse=True)

        self.new_recon_img = self.incomplete_img + self.recon_img * self.mask

        self.D_real_gan_logits = self.discriminator(self.input, self.local_input_left, self.local_input_right,
                                                    self.angle_invar_left_real, self.angle_invar_right_real, reuse=False)
        self.D_fake_gan_logits = self.discriminator(self.new_recon_img, self.local_recon_img_left, self.local_recon_img_right,
                                                    self.angle_invar_left_real, self.angle_invar_right_real, reuse=True)

        if self.loss_type == 0:
            self.d_gan_loss = self.loss_hinge_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_hinge_gen(self.D_fake_gan_logits)
        elif self.loss_type == 1:
            self.d_gan_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.loss_gen(self.D_fake_gan_logits)
        elif self.loss_type == 2:
            self.d_gan_loss = self.d_lsgan_loss(self.D_real_gan_logits, self.D_fake_gan_logits)
            self.g_gan_loss = self.g_lsgan_loss(self.D_fake_gan_logits)

        self.percep_loss = tf.reduce_mean(tf.square(self.angle_invar_left_real - self.angle_invar_left_recon)) + \
                       tf.reduce_mean(tf.square(self.angle_invar_right_real - self.angle_invar_right_recon))

        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.new_recon_img - self.input), axis=[1, 2, 3])
                                         / (35 * 70 * self.channel))

        self.D_loss = self.d_gan_loss
        self.G_loss = self.g_gan_loss + self.lam_recon * self.recon_loss + self.lam_percep * self.percep_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.ed_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]

        print("d_vars", len(self.d_vars))
        print("ed_vars", len(self.ed_vars))
        print("e_vars", len(self.e_vars))

        self.saver = tf.train.Saver()
        self.pretrain_saver = tf.train.Saver(self.e_vars)


        for k, v in self.log_vars:
            tf.summary.scalar(k, v)


    def crop_and_resize(self,input,boxes):
       return tf.image.crop_and_resize(input, boxes=boxes,box_ind=range(0, self.batch_size),
                                       crop_size=[self.output_size / 2, self.output_size / 2])



    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    def loss_hinge_dis(self, d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(self, d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    def d_lsgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(self, d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    def test(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            print('Load checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
            else:
                print 'Do not exists any checkpoint,Load Failed!'
                exit()

            _,_,_, testbatch, testmask = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for j in range(self.batch_num):
                real_test_batch, real_eye_pos = sess.run([testbatch, testmask])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)
                f_d = {self.input: real_test_batch, self.mask: batch_masks,
                       self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos}

                mask, test_incomplete_img, test_recon_img, new_test_recon_img = \
                    sess.run ([self.mask,self.incomplete_img, self.recon_img, self.new_recon_img], feed_dict=f_d)
                test_output_concat = np.concatenate([real_test_batch, mask, test_incomplete_img, test_recon_img,
                                                     new_test_recon_img], axis=0)
                save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                        '{}/{:02d}_test_output.jpg'.format(self.result_dir, j))

            coord.request_stop()
            coord.join(threads)



    def train(self):

        opti_D = tf.train.AdamOptimizer(self.d_learning_rate * self.lr_decay,beta1=self.beta1, beta2=self.beta2).\
                                        minimize(loss=self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(self.g_learning_rate * self.lr_decay,beta1=self.beta1, beta2=self.beta2).\
                                        minimize(loss=self.G_loss, var_list=self.ed_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)

            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                try:
                    self.pretrain_saver.restore(sess, os.path.join(self.pretrain_model_dir,
                                                               'model_{:06d}.ckpt'.format(self.pretrain_model_index)))
                except Exception as e:
                    print(" Self-Guided Model path may not be correct")

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = self.start_step
            lr_decay = self.lr_init

            print("Start read dataset")

            image_path, train_images, train_eye_pos, test_images, test_eye_pos = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start entering the looping")
            real_test_batch, real_test_pos = sess.run([test_images, test_eye_pos])

            while step <= self.max_iters:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                real_batch_image_path, real_batch_image, real_eye_pos = sess.run([image_path,train_images, train_eye_pos])
                batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_eye_pos)


                f_d = {self.input: real_batch_image, self.mask: batch_masks,
                       self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos,
                       self.lr_decay: lr_decay}

                # optimize D
                sess.run(opti_D, feed_dict=f_d)
                # optimize G
                sess.run(opti_G, feed_dict=f_d)

                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:

                    output_loss = sess.run([self.D_loss, self.G_loss, self.lam_recon * self.recon_loss,
                                            self.lam_percep * self.percep_loss], feed_dict=f_d)

                    print("step %d D_loss=%.8f, G_loss=%.4f, Recon_loss=%.4f, Percep_loss=%.4f, lr_decay=%.4f" %
                                    (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], lr_decay))

                if np.mod(step, 2000) == 0:
                    train_output_img = sess.run([self.local_input_left, self.local_input_right, self.incomplete_img,
                                                 self.recon_img,self.new_recon_img, self.local_recon_img_left,
                                                 self.local_recon_img_right], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)
                    #for test
                    f_d = {self.input: real_test_batch, self.mask: batch_masks,
                           self.input_left_labels: batch_left_eye_pos, self.input_right_labels: batch_right_eye_pos,
                           self.lr_decay: lr_decay}

                    test_output_img = sess.run([self.incomplete_img, self.recon_img, self.new_recon_img], feed_dict=f_d)

                    output_concat = np.concatenate([real_batch_image, train_output_img[2], train_output_img[3],
                                                    train_output_img[4]], axis=0)
                    local_output_concat = np.concatenate([train_output_img[0], train_output_img[1], train_output_img[5],
                                                         train_output_img[6]], axis=0)
                    test_output_concat = np.concatenate([real_test_batch, test_output_img[0], test_output_img[2],
                                                         test_output_img[1]], axis=0)

                    save_images(local_output_concat, [local_output_concat.shape[0] / self.batch_size, self.batch_size],
                                            '{}/{:02d}_local_output.jpg'.format(self.sample_dir, step))
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_dir, step))
                    save_images(test_output_concat, [test_output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_test_output.jpg'.format(self.sample_dir, step))

                if np.mod(step, 20000) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.model_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.model_dir, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print "Model saved in file: %s" % save_path

    def discriminator(self, incom_x, local_x_left, local_x_right, guided_fp_left, guided_fp_right, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            # Global Discriminator Dg
            x = incom_x

            for i in range(6):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                print output_dim
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_1_{}'.format(i)))

            x = tf.reshape(x, shape=[self.batch_size, -1])
            ful_global = fully_connect(x, output_size=output_dim, use_sp=self.use_sp, scope='dis_fu1')

            # Local Discriminator Dl
            x = tf.concat([local_x_left, local_x_right], axis=3)

            for i in range(5):
                output_dim = np.minimum(16 * np.power(2, i+1), 256)
                x = lrelu(conv2d(x, output_dim=output_dim, use_sp=self.use_sp, name='dis_conv_2_{}'.format(i)))

            x = tf.reshape(x, shape=[self.batch_size, -1])
            ful_local = fully_connect(x, output_size=output_dim*2, use_sp=self.use_sp, scope='dis_fu2')

            # Concatenation
            ful = tf.concat([ful_global, ful_local, guided_fp_left, guided_fp_right], axis=1)
            ful = tf.nn.relu(fully_connect(ful, output_size=512, use_sp=self.use_sp, scope='dis_fu4'))
            gan_logits = fully_connect(ful, output_size=1, use_sp=self.use_sp, scope='dis_fu5')

            return gan_logits

    def generator(self, input_x, img_mask, guided_fp_left, guided_fp_right, use_sp=False, reuse=False):

        with tf.variable_scope("generator") as scope:

            if reuse == True:
                scope.reuse_variables()

            x = tf.concat([input_x, img_mask], axis=3)
            for i in range(6):
                c_dim = np.minimum(16 * np.power(2, i), 256)
                if i == 0:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=7, k_h=7, d_w=1, d_h=1, use_sp=use_sp,
                                             name='conv_{}'.format(i)),scope='conv_IN_{}'.format(i)))
                    print(x)
                else:
                    x = tf.nn.relu(
                        instance_norm(conv2d(x, output_dim=c_dim, k_w=4, k_h=4, d_w=2, d_h=2, use_sp=use_sp,
                                             name='conv_{}'.format(i)),scope='conv_IN_{}'.format(i)))

            bottleneck = tf.reshape(x, shape=[self.batch_size, -1])
            bottleneck = fully_connect(bottleneck, output_size=256, use_sp=use_sp, scope='FC1')
            bottleneck = tf.concat([bottleneck, guided_fp_left, guided_fp_right], axis=1)

            de_x = tf.nn.relu(fully_connect(bottleneck, output_size=256*8*8, use_sp=use_sp, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.batch_size, 8, 8, 256])

            for i in range(5):
                c_dim = np.maximum(256 / np.power(2, i), 16)
                output_dim = 16 * np.power(2, i)
                print de_x
                de_x = tf.nn.relu(instance_norm(de_conv(de_x, output_shape=[self.batch_size, output_dim, output_dim, c_dim], use_sp=use_sp,
                                                            name='deconv_{}'.format(i)), scope='deconv_IN_{}'.format(i)))
            recon_img1 = conv2d(de_x, output_dim=3, k_w=7, k_h=7, d_h=1, d_w=1, use_sp=use_sp, name='output_conv')

            return tf.nn.tanh(recon_img1)

    # DO NOT CHANGE SCOPE NAME
    def encode(self, x, reuse=False):

        with tf.variable_scope("encode") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x, output_dim=32, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=64, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))
            conv4 = tf.nn.relu(
                instance_norm(conv2d(conv3, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c4'), scope='e_in4'))

            bottleneck = tf.reshape(conv4, [self.batch_size, -1])
            content = fully_connect(bottleneck, output_size=128, scope='e_ful1')

            return content

    def get_Mask_and_pos(self, eye_pos, flag=0):

        eye_pos = eye_pos
        batch_mask = []
        batch_left_eye_pos = []
        batch_right_eye_pos = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            left_eye_pos = []
            right_eye_pos = []

            if flag == 0:

                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_eye_pos[1] - 5
                down_scale = current_eye_pos[1] + 30
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_eye_pos[0] - 35
                down_scale = current_eye_pos[0] + 35
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
                left_eye_pos.append(float(l1_1)/self.output_size)
                left_eye_pos.append(float(l1_2)/self.output_size)
                left_eye_pos.append(float(u1_1)/self.output_size)
                left_eye_pos.append(float(u1_2)/self.output_size)

                scale = current_eye_pos[3] - 5
                down_scale = current_eye_pos[3] + 30
                l2_1 = int(scale)
                u2_1 = int(down_scale)

                scale = current_eye_pos[2] - 35
                down_scale = current_eye_pos[2] + 35
                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

                right_eye_pos.append(float(l2_1) / self.output_size)
                right_eye_pos.append(float(l2_2) / self.output_size)
                right_eye_pos.append(float(u2_1) / self.output_size)
                right_eye_pos.append(float(u2_2) / self.output_size)

            batch_mask.append(mask)
            batch_left_eye_pos.append(left_eye_pos)
            batch_right_eye_pos.append(right_eye_pos)

        return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)







