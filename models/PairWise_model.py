import pdb

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(2023)
import numpy as np


class Base_CF(object):
    def __init__(self, args):
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.latent_dim = args.latent_dim
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.nor_initializer = tf.random_normal_initializer(stddev=0.01)
        # self.xa_initializer = tf.contrib.layers.xavier_initializer()
        # pre_emb = np.load('/home/yangyonghui/myworks/IBRec/run/node2vec/node_embeddings.npy')
        # self.pre_emb = np.load('/home/yangyonghui/myworks/GHCF-main/Data/Taobao/node_embeddings3.npy')
        # self.pre_emb = np.load('/home/yangyonghui/myworks/GHCF-main/Data/Beibei/node_embeddings_0.005.npy')
        self.pre_emb = np.load('/home/yangyonghui/myworks/GHCF-main/Data/tmall31/node_embeddings_all_0.005.npy')
        # pdb.set_trace()

        with tf.name_scope('create_variables'):
            # self.user_latent_emb = tf.Variable(self.xa_initializer([self.num_user, self.latent_dim]), name='user_latent_emb')
            # self.item_latent_emb = tf.Variable(self.xa_initializer([self.num_item, self.latent_dim]), name='item_latent_emb')
            self.user_latent_emb = tf.Variable(self.nor_initializer([self.num_user, self.latent_dim]), name='user_latent_emb')
            self.item_latent_emb = tf.Variable(self.nor_initializer([self.num_item, self.latent_dim]), name='item_latent_emb')
            # self.user_latent_emb = tf.Variable(initial_value=self.pre_emb[:args.num_user],
            #                                    trainable=True, name='user_latent_emb', dtype=tf.float32)
            # self.item_latent_emb = tf.Variable(initial_value=self.pre_emb[args.num_user:],
            #                                    trainable=True, name='item_latent_emb', dtype=tf.float32)

        with tf.name_scope('input_data'):
            self.users = tf.compat.v1.placeholder(tf.int32, [None])
            self.pos_items = tf.compat.v1.placeholder(tf.int32, [None])
            self.neg_items = tf.compat.v1.placeholder(tf.int32, [None])

        ### for visualization
        with tf.name_scope('train_loss'):
            self.train_loss = tf.compat.v1.placeholder(tf.float32)
            tf.summary.scalar('loss', self.train_loss)
            self.train_mf_loss = tf.compat.v1.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_loss'))
        with tf.name_scope('evaluate_metrics'):
            self.recall = tf.compat.v1.placeholder(tf.float32)
            tf.summary.scalar('recall', self.recall)
            self.ndcg = tf.compat.v1.placeholder(tf.float32)
            tf.summary.scalar('ndcg', self.ndcg)
        self.merged_evaluate = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), 'evaluate_metrics')


    def compute_bpr_loss(self, user_emb, item_emb, sum_Flag=False):
        batch_user_emb = tf.nn.embedding_lookup(user_emb, self.users)
        batch_pos_item_emb = tf.nn.embedding_lookup(item_emb, self.pos_items)
        batch_neg_item_emb = tf.nn.embedding_lookup(item_emb, self.neg_items)
        user_reg_emb = tf.nn.embedding_lookup(self.user_latent_emb, self.users)
        item_reg_pos_emb = tf.nn.embedding_lookup(self.item_latent_emb, self.pos_items)
        item_reg_neg_emb = tf.nn.embedding_lookup(self.item_latent_emb, self.neg_items)
        pos_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_pos_item_emb), 1, keepdims=True)
        neg_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_neg_item_emb), 1, keepdims=True)
        auc = tf.reduce_mean(tf.cast(pos_scores > neg_scores, tf.float32))
        if sum_Flag:
            # regularization = tf.nn.l2_loss(user_reg_emb) + tf.nn.l2_loss(item_reg_pos_emb) + tf.nn.l2_loss(item_reg_neg_emb)
            regularization = tf.nn.l2_loss(batch_pos_item_emb) + tf.nn.l2_loss(batch_user_emb) + tf.nn.l2_loss(batch_neg_item_emb)
            bpr_loss = -tf.reduce_sum(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores), 1e-9, 1.0)))
            regu_loss = 1/2 * self.l2_reg * (regularization)
        else:
            regularization = tf.nn.l2_loss(user_reg_emb) + tf.nn.l2_loss(item_reg_pos_emb) + tf.nn.l2_loss(item_reg_neg_emb)
            # regularization = tf.nn.l2_loss(batch_user_emb) + tf.nn.l2_loss(batch_pos_item_emb) + tf.nn.l2_loss(batch_neg_item_emb)
            bpr_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores), 1e-9, 1.0)))
            regu_loss = 1/2 * self.l2_reg * (regularization / self.batch_size)
        return bpr_loss, regu_loss, auc



    def Softmax_loss_batch(self, user_emb, item_emb, temp=0.2):
        batch_user_emb = tf.nn.embedding_lookup(user_emb, self.users)
        batch_item_emb = tf.nn.embedding_lookup(item_emb, self.pos_items)
        batch_user_emb = tf.nn.l2_normalize(batch_user_emb, 1)
        batch_item_emb = tf.nn.l2_normalize(batch_item_emb, 1)
        neg_scores = tf.exp(tf.matmul(batch_user_emb, tf.transpose(tf.nn.l2_normalize(batch_item_emb, 1))) / temp)
        neg_scores = tf.reduce_sum(neg_scores, 1, keepdims=True)
        pos_scores = tf.exp(tf.reduce_sum(tf.multiply(batch_user_emb, batch_item_emb), 1, keepdims=True) / temp)
        loss = -tf.log(pos_scores / neg_scores)
        return loss



    def invariant_loss(self, user_emb, item_emb):
        batch_user_emb = tf.nn.embedding_lookup(user_emb, self.users)
        batch_pos_item_emb = tf.nn.embedding_lookup(item_emb, self.pos_items)
        batch_neg_item_emb = tf.nn.embedding_lookup(item_emb, self.neg_items)
        pos_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_pos_item_emb), 1, keepdims=True)
        neg_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_neg_item_emb), 1, keepdims=True)
        bpr_loss = -tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores), 1e-9, 1.0))
        return bpr_loss


    # def compute_margin_loss(self, user_emb, item_emb):
    #     batch_user_emb = tf.nn.embedding_lookup(user_emb, self.users)
    #     batch_pos_item_emb = tf.nn.embedding_lookup(item_emb, self.pos_items)
    #     batch_neg_item_emb = tf.nn.embedding_lookup(item_emb, self.neg_items)
    #     user_reg_emb = tf.nn.embedding_lookup(self.user_latent_emb, self.users)
    #     item_reg_pos_emb = tf.nn.embedding_lookup(self.item_latent_emb, self.pos_items)
    #     item_reg_neg_emb = tf.nn.embedding_lookup(self.item_latent_emb, self.neg_items)
    #     pos_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_pos_item_emb), 1, keepdims=True)
    #     neg_scores = tf.reduce_sum(tf.multiply(batch_user_emb, batch_neg_item_emb), 1, keepdims=True)
    #     rec_loss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (pos_scores - negPred)))