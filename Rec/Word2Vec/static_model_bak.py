# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import math
import numpy as np


class Model(object):
    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()

    def init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.neg_num = self.config.get("hyper_parameters.neg_num")
        self.with_shuffle_batch = self.config.get(
            "hyper_parameters.with_shuffle_batch")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.decay_steps = self.config.get(
            "hyper_parameters.optimizer.decay_steps")
        self.decay_rate = self.config.get(
            "hyper_parameters.optimizer.decay_rate")

    def input_data(self, is_infer=False, **kwargs):
        if is_infer:
            analogy_a = paddle.static.data(
                name="analogy_a", shape=[None], dtype='int64')
            analogy_b = paddle.static.data(
                name="analogy_b", shape=[None], dtype='int64')
            analogy_c = paddle.static.data(
                name="analogy_c", shape=[None], dtype='int64')
            analogy_d = paddle.static.data(
                name="analogy_d", shape=[None], dtype='int64')
            analogy_f = paddle.static.data(
                name="analogy_f", shape=[None], dtype='int64')
            return [analogy_a, analogy_b, analogy_c, analogy_d, analogy_f]

        input_word = paddle.static.data(
            name="input_word", shape=[None, 1], dtype='int64')
        true_word = paddle.static.data(
            name='true_label', shape=[None, 1], dtype='int64')
        if self.with_shuffle_batch:
            return [input_word, true_word]

        neg_word = paddle.static.data(
            name="neg_label", shape=[None, self.neg_num], dtype='int64')
        return [input_word, true_word, neg_word]

    def net(self, inputs, is_infer=False):
        if is_infer:
            return self.infer_net(inputs)

        word2vec_model = Word2VecLayer(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            self.neg_num,
            emb_name="emb",
            emb_w_name="emb_w",
            emb_b_name="emb_b")
        true_logits, neg_logits = word2vec_model.forward(inputs)

        label_ones = paddle.full(
            shape=[paddle.shape(true_logits)[0], 1], fill_value=1.0)
        label_zeros = paddle.full(
            shape=[paddle.shape(true_logits)[0], self.neg_num], fill_value=0.0)

        true_logits = paddle.nn.functional.sigmoid(true_logits)
        true_xent = paddle.nn.functional.binary_cross_entropy(true_logits,
                                                              label_ones)
        neg_logits = paddle.nn.functional.sigmoid(neg_logits)
        neg_xent = paddle.nn.functional.binary_cross_entropy(neg_logits,
                                                             label_zeros)
        cost = paddle.add(true_xent, neg_xent)
        avg_cost = paddle.mean(x=cost)

        self.infer_target_var = avg_cost
        self.cost = avg_cost
        self.metrics["LOSS"] = avg_cost

        return self.metrics

    def minimize(self, strategy=None):
        lr = float(self.config.get("hyper_parameters.optimizer.learning_rate"))
        decay_rate = float(self.config.get(
            "hyper_parameters.optimizer.decay_rate"))
        decay_steps = int(self.config.get(
            "hyper_parameters.optimizer.decay_steps"))

        # single
        if strategy == None:
            optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=lr,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True))
        else:
            sync_mode = self.config.get("static_benchmark.sync_mode")
            # geo
            if sync_mode == "geo":
                decay_steps = int(decay_steps / fleet.worker_num())
                optimizer = fluid.optimizer.SGD(
                    learning_rate=fluid.layers.exponential_decay(
                        learning_rate=lr,
                        decay_steps=decay_steps,
                        decay_rate=decay_rate,
                        staircase=True))

            # async sync heter
            if sync_mode in ["async", "sync", "heter"]:
                scheduler = paddle.optimizer.lr.ExponentialDecay(
                    learning_rate=lr,
                    gamma=decay_rate,
                    verbose=True)
                optimizer = fluid.optimizer.SGD(scheduler)
                strategy.a_sync_configs = {"lr_decay_steps": decay_steps}

            optimizer = fleet.distributed_optimizer(optimizer, strategy)

        optimizer.minimize(self.cost)


class Word2VecLayer(nn.Layer):
    def __init__(self, sparse_feature_number, emb_dim, neg_num, emb_name,
                 emb_w_name, emb_b_name):
        super(Word2VecLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.emb_dim = emb_dim
        self.neg_num = neg_num
        self.emb_name = emb_name
        self.emb_w_name = emb_w_name
        self.emb_b_name = emb_b_name

        init_width = 0.5 / self.emb_dim
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.emb_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_name,
                initializer=paddle.nn.initializer.Uniform(-init_width,
                                                          init_width)))

        self.embedding_w = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.emb_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_w_name,
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.embedding_b = paddle.nn.Embedding(
            self.sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_b_name,
                initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, inputs):
        input_emb = self.embedding(inputs[0])
        true_emb_w = self.embedding_w(inputs[1])
        true_emb_b = self.embedding_b(inputs[1])
        neg_emb_w = self.embedding_w(inputs[2])
        neg_emb_b = self.embedding_b(inputs[2])

        with fluid.device_guard("gpu"):
            input_emb = paddle.squeeze(x=input_emb, axis=[1])
            true_emb_w = paddle.squeeze(x=true_emb_w, axis=[1])
            true_emb_b = paddle.squeeze(x=true_emb_b, axis=[1])

            neg_emb_b_vec = paddle.reshape(neg_emb_b, shape=[-1, self.neg_num])

            true_logits = paddle.add(x=paddle.sum(x=paddle.multiply(
                x=input_emb, y=true_emb_w),
                axis=1,
                keepdim=True),
                y=true_emb_b)

            input_emb_re = paddle.reshape(
                input_emb, shape=[-1, 1, self.emb_dim])
            neg_matmul = paddle.matmul(
                input_emb_re, neg_emb_w, transpose_y=True)
            neg_matmul_re = paddle.reshape(
                neg_matmul, shape=[-1, self.neg_num])
            neg_logits = paddle.add(x=neg_matmul_re, y=neg_emb_b_vec)

        return true_logits, neg_logits
