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

from abc import ABCMeta, abstractmethod


class tensorflowModel:
    """
    Abstract class for tensorflow model.
    This class defines the API interface for user to create a tensorflow estimator model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_input_fn, eval_input_fn):
        pass

    @abstractmethod
    def evaluate(self, eval_input_fn):
        pass

    @abstractmethod
    def predict(self, test_input_fn):
        pass

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