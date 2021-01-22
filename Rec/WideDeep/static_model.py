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
import math
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet


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
        self.sparse_inputs_slot = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")
        self.adam_lazy_mode = self.config.get(
            "hyper_parameters.optimizer.adam_lazy_mode")

    def input_data(self):
        dense_input = fluid.layers.data(name="dense_input",
                                        shape=[self.dense_input_dim],
                                        dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(name="C" + str(i),
                              shape=[1],
                              lod_level=1,
                              dtype="int64") for i in range(1, self.sparse_inputs_slot)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, is_infer=False):
        self.sparse_inputs = inputs[1:-1]
        self.dense_input = inputs[0]
        self.label_input = inputs[-1]

        wide_deep_model = WideDeepLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, self.sparse_inputs_slot-1, self.fc_sizes)

        pred = wide_deep_model(self.sparse_inputs, self.dense_input)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                    label=self.label_input,
                                                    num_thresholds=2**12,
                                                    slide_steps=20)

        if is_infer:
            self._infer_results["AUC"] = auc
            return

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost
        self.infer_target_var = auc
        return {'cost': avg_cost, 'auc': auc}

    def minimize(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            self.learning_rate, lazy_mode=self.adam_lazy_mode)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)

    def infer_net(self):
        pass


class WideDeepLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(WideDeepLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.wide_part = paddle.nn.Linear(
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):
        # wide part
        wide_output = self.wide_part(dense_inputs)

        # deep part
        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)

        deep_output = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)
        for n_layer in self._mlp_layers:
            deep_output = n_layer(deep_output)

        prediction = paddle.add(x=wide_output, y=deep_output)
        pred = F.sigmoid(prediction)
        return pred
