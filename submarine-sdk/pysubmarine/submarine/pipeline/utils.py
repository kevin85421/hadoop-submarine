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

import tensorflow as tf
import json
import os
from submarine.utils.env import check_env_exists


def merge_json(path, defaultParams):
    """
    Merge parameters json parameter into default parameters
    :param path: The json file that specifies the model parameters.
    :type path: String
    :param defaultParams: default parameters for model parameters
    :type path: Dictionary
    :return:
    """
    with open(path) as json_data:
        params = json.load(json_data)
    return merge_dicts(params, defaultParams)


def merge_dicts(params, defaultParams):
    """
    Merge two dictionary
    :param params: parameters will be merged
    :type params: Dictionary
    :param defaultParams: default parameters for model parameters
    :type params: Dictionary
    :return:
    """
    if params is None:
        return defaultParams
    merge_params = defaultParams.copy()
    merge_params.update(params)
    return merge_params


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance with appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    # GPU limit: TensorFlow by default allocates all GPU memory:
    # If multiple workers run in same host you may see OOM errors:
    # Use as workaround if not using Hadoop 3.1
    # Change percentage accordingly:
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] \
            and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps.
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps.
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(  # gpu_options=gpu_options,
                                  device_filters=[
                                      '/job:ps',
                                      '/job:worker/task:%d' % tf_config['task'][
                                          'index']
                                  ])
    return None


def get_TFConfig(params):
    """
    Get TF_CONFIG to run local or distributed training, If user don't set TF_CONFIG environment
     variables, by default set local mode
    :param params: model parameters that contain total number of gpu or cpu the model
    intends to use
    :type params: Dictionary
    :return: The class specifies the configurations for an Estimator run
    """
    if params['mode'] == 'local':  # local mode
        tf_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                device_count={'GPU': params['num_gpu'], 'CPU': params['num_threads']}),
            log_step_count_steps=params['log_steps'], save_summary_steps=params['log_steps'])
    elif params['mode'] == 'submarine':  # TODO distributed mode
        tf_config = tf.estimator.RunConfig(
            experimental_distribute=tf.contrib.distribute.DistributeConfig(
                train_distribute=tf.contrib.distribute.ParameterServerStrategy(),
                eval_distribute=tf.contrib.distribute.MirroredStrategy()),
            session_config=_get_session_config_from_env_var(),
            save_summary_steps=params['log_steps'],
            log_step_count_steps=params['log_steps'])
    else:
        raise ValueError("mode should be local or distributed")
    return tf_config
