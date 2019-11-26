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
from submarine.utils.env import check_env_exists


def merge_dicts(params, defaultParams):
    """
    Merge two dictionary
    :param params: parameters will be merged
    :type params: Dictionary
    :param defaultParams: default parameters in dict
    :type params: Dictionary
    :return:
    """
    if params is None:
        return defaultParams
    merge_params = params.copy()   # start with x's keys and values
    merge_params.update(defaultParams)    # modifies z with y's keys and values & returns None
    return merge_params


def get_TFConfig(params):
    """
    Get TF_CONFIG to run local or distributed training, If user don't set TF_CONFIG environment variables, by
    default set local mode
    :param params: model parameters that contain total number of gpu or cpu the model intends to use
    :return: The class specifies the configurations for an Estimator run
    """
    config = tf.estimator.RunConfig()
    if check_env_exists('TF_CONFIG'):
        return tf.estimator.RunConfig()
    if params['mode'] == 'local':  # local mode
        config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'GPU': params['num_gpu'], 'CPU': params['num_threads']}),
            log_step_count_steps=params['log_steps'], save_summary_steps=params['log_steps'])
    elif params['mode'] == 'distributed':  # TODO distributed mode
        config = None
    else:
        raise ValueError("mode should be local or distributed")
    return config
