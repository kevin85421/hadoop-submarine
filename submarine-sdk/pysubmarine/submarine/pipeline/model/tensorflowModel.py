# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta


class tensorflowModel:
    """
    Abstract class for tensorflow model.
    This class defines the API interface for user to create a tensorflow estimator model.
    """

    __metaclass__ = ABCMeta

    def __init__(self, model_dir, config=None, model_params=None, json_path=None):
        """
        Create a tensorflow DeepFM model
        :param model_dir: A model directory for saving model
        :param config: The class specifies the configurations for an Estimator run
        :param model_params: defines the different
        parameters of the model, features, preprocessing and training
        :param json_path: The json file that specifies the model parameters.
        """

    def train(self, train_input_fn, eval_input_fn):
        """
        Trains a pre-defined tensorflow estimator model with given training data
        :param train_input_fn: A function that provides input data for training.
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: None
        """
        pass

    def evaluate(self, eval_input_fn):
        """
        Evaluates a pre-defined tensorflow estimator model with given evaluate data
        :param eval_input_fn: A function that provides input data for evaluating.
        :return: None
        """
        pass

    def predict(self, test_input_fn):
        """
        Yields predictions with given features.
        :param test_input_fn: A function that constructs the features.
         Prediction continues until input_fn raises an end-of-input exception
        :return: Evaluated values of predictions tensors.
        """

    def setParameter(self, key, value):
        """
        set model parameter
        :param key: a key of model parameters
        :param value: a value of model parameters
        :return: None
        """
        pass

    def getParameter(self, key):
        """
        set model parameter
        :param key: a key of model parameters
        :return: a value of model parameters
        """
        pass
