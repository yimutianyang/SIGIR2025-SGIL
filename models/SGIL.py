import tensorflow as tf
import os, pdb
import sys
sys.path.append('./models/')
from PairWise_model import Base_CF
import numpy as np
from collections import defaultdict


class SGIL(Base_CF):
    def __init__(self, args, dataset):
        super(SGIL, self).__init__(args)
        self.gcn_layer = args.gcn_layer
        self.num_envs = args.num_envs
        self.penalty_coff = args.penalty_coff
        self.edge_bias = args.edge_bias
        self.adj_indices, self.adj_values, self.adj_shape = dataset.convert_csr_to_sparse_tensor_inputs(dataset.uu_i_matrix)
        self.adj_matrix = tf.SparseTensor(self.adj_indices, self.adj_values, self.adj_shape)
        self.env_generators = {}
        for k in range(self.num_envs):
            self.env_generators[k] = self.generator_model()
        self._build_graph()


    def generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dropout(rate=0.5))
        # model.add(tf.keras.layers.Dense(int(self.latent_dim/2), activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dropout(rate=0.5))
        model.add(tf.keras.layers.Dense(1, activation=None))
        return model


    def graph_generaotr_V2(self, ego_emb, k):
        '''
        all edge generator
        '''
        row = self.adj_matrix.indices[:, 0]
        col = self.adj_matrix.indices[:, 1]
        row_emb = tf.nn.embedding_lookup(ego_emb, row)
        col_emb = tf.nn.embedding_lookup(ego_emb, col)
        cat_emb = tf.concat([row_emb, col_emb], axis=1)  # [n, 2d]
        logit = self.env_generators[k](cat_emb)
        logit = tf.reshape(logit, [-1, ])
        # bias = 0.0 + 0.0001
        # eps = (bias - (1 - bias)) * tf.random_uniform(logit.shape) + (1 - bias)
        eps = tf.random_uniform(logit.shape) #
        mask_gate_input = tf.log(eps) - tf.log(1 - eps)
        mask_gate_input = (logit+mask_gate_input) / 0.2
        mask_gate_input = tf.nn.sigmoid(mask_gate_input) + self.edge_bias #self.edge_bias
        masked_values = tf.multiply(self.adj_matrix.values, mask_gate_input)
        # masked_values = tf.stop_gradient(masked_values)
        masked_adj_matrix = tf.SparseTensor(self.adj_matrix.indices, masked_values, self.adj_matrix.shape)
        return masked_adj_matrix

        # degree = tf.sparse.reduce_sum(masked_adj_matrix, axis=1)
        # degree = tf.cast(degree, dtype=tf.float32)
        # degree_inv_sqrt = tf.math.pow(degree, -0.5)
        # degree_inv_sqrt = tf.where(tf.math.is_inf(degree_inv_sqrt), tf.zeros_like(degree_inv_sqrt), degree_inv_sqrt)
        # degree_inv_sqrt = tf.diag(degree_inv_sqrt)
        # # Create diagonal sparse matrix for D^(-1/2)
        # degree_inv_sqrt_indices = tf.stack([tf.range(self.num_user+self.num_item, dtype=tf.int64),
        #                                     tf.range(self.num_user+self.num_item, dtype=tf.int64)], axis=1)
        # degree_inv_sqrt_values = degree_inv_sqrt
        # degree_inv_sqrt_shape = self.adj_shape
        # degree_inv_sqrt_matrix = tf.SparseTensor(
        #     indices=degree_inv_sqrt_indices,
        #     values=degree_inv_sqrt_values,
        #     dense_shape=degree_inv_sqrt_shape
        # )
        # normalized_adj_dense = \
        #     tf.matmul(degree_inv_sqrt, tf.sparse.to_dense(masked_adj_matrix)) # D^-0.5 * A
        # normalized_adj_dense = tf.matmul(normalized_adj_dense, degree_inv_sqrt)
        #
        # indices = tf.where(tf.not_equal(normalized_adj_dense, 0))
        # values = tf.gather_nd(normalized_adj_dense, indices)
        # shape = tf.shape(normalized_adj_dense, out_type=tf.int64)
        # normalized_adj_sparse = tf.SparseTensor(indices, values, shape)
        # return normalized_adj_sparse



    def _create_masked_lightgcn_emb(self, ego_emb, k):
        '''
        compute subgraph embedding
        '''
        masked_adj_matrix = self.graph_generaotr_V2(ego_emb, k)
        all_emb_masked = [ego_emb]
        for _ in range(self.gcn_layer):
            cur_emb = tf.sparse.sparse_dense_matmul(masked_adj_matrix, all_emb_masked[-1])
            all_emb_masked.append(cur_emb)
        all_emb_masked = tf.stack(all_emb_masked, axis=1)
        mean_emb_masked = tf.reduce_mean(all_emb_masked, axis=1, keepdims=False)
        return mean_emb_masked


    def _build_graph(self):
        with tf.name_scope('forward'):
            ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
            embedding_list = []
            for k in range(self.num_envs):
                embedding_list.append(self._create_masked_lightgcn_emb(ego_emb, k))
            all_emb_envs = tf.stack(embedding_list, axis=0)
            avg_emb_envs = tf.reduce_mean(all_emb_envs, axis=0)
            self.user_emb, self.item_emb = tf.split(avg_emb_envs, [self.num_user, self.num_item], axis=0)

        with tf.name_scope('optimization'):
            generator_param = []
            for k in range(self.num_envs):
                generator_param.extend(self.env_generators[k].trainable_variables)
            loss_list = []
            for k in range(self.num_envs):
                cur_user_emb, cur_item_emb = tf.split(embedding_list[k], [self.num_user, self.num_item], axis=0)
                loss_list.append(self.Softmax_loss_batch(cur_user_emb, cur_item_emb))
            self.loss = tf.reduce_mean(tf.concat(loss_list, axis=1))
            _, var = tf.nn.moments(tf.concat(loss_list, axis=1), axes=1)
            self.penalty = tf.reduce_sum(var) * self.penalty_coff
            self.l2_loss = self.L2_regularizer([self.user_latent_emb, self.item_latent_emb]) + \
                           self.L2_regularizer(generator_param)
            # self.loss += self.l2_loss
            self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss + self.penalty)
            self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(-self.penalty, var_list=generator_param)


    def L2_regularizer(self, para_list):
        loss = tf.nn.l2_loss(para_list[0]) * self.l2_reg
        for k in range(len(para_list)):
            loss += tf.nn.l2_loss(para_list[k]) * self.l2_reg
        return loss / self.batch_size
