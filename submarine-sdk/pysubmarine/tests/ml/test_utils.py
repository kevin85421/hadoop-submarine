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

import pytest
import json
import os
from submarine.ml.utils import sanity_checks, merge_json, merge_dicts,\
    get_from_registry, get_TFConfig, json_validate


@pytest.fixture(scope="function")
def output_json_filepath():
    params = {"learning_rate": 0.05}
    path = '/tmp/data.json'
    with open(path, 'w') as f:
        json.dump(params, f)
    return path


def test_sanity_checks():
    params = {"learning rate": 0.05}
    with pytest.raises(AssertionError, match="Does not define any input parameters"):
        sanity_checks(params)

    params.update({'input': {'train_data': '/tmp/train.csv'}})
    with pytest.raises(AssertionError, match="Does not define any input type"):
        sanity_checks(params)

    params.update({'input': {'type': 'libsvm'}})
    with pytest.raises(AssertionError, match="Does not define any output parameters"):
        sanity_checks(params)

    params.update({'output': {'save_model_dir': '/tmp/save'}})


def test_merge_json(output_json_filepath):
    defaultParams = {"learning_rate": 0.08, "embedding_size": 256}
    params = merge_json(output_json_filepath, defaultParams)
    assert params['learning_rate'] == 0.05
    assert params['embedding_size'] == 256


def test_merge_dicts():
    params = {"learning_rate": 0.05}
    defaultParams = {"learning_rate": 0.08, "embedding_size": 256}
    final = merge_dicts(params, defaultParams)
    assert final['learning_rate'] == 0.05
    assert final['embedding_size'] == 256


def test_get_from_registry():
    registry = {'model': 'xgboost'}
    val = get_from_registry('MODEL', registry)
    assert val == 'xgboost'

    with pytest.raises(ValueError):
        get_from_registry('test', registry)


def test_get_TFConfig():
    params = {'training': {'mode': 'test'}}
    with pytest.raises(ValueError, match="mode should be local or distributed"):
        get_TFConfig(params)

    # run local training
    params.update({'training': {'mode': 'local', 'num_gpu': 0, 'num_threads': 4, 'log_steps': 10}})
    get_TFConfig(params)

    # run distributed training
    params.update({'training': {'mode': 'distributed', 'log_steps': 10}})
    get_TFConfig(params)

def test_json_validate():
    with open('testcase/test.json', 'r') as f:
        defaultParams = json.load(f)
    # valid cases
    # [CASE1]: remove valid secondKey 
    with open('testcase/test2.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is True
    # [CASE2]: change the values of firstKeys 
    with open('testcase/test5.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is True
    # [CASE3]: change the values of secondKeys
    with open('testcase/test6.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is True
    # [CASE4]: remove all secondKeys
    with open('testcase/test7.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is True
    # [CASE5]: empty JSON file
    with open('testcase/test9.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is True

    # invalid cases
    # [CASE1]: Change the data type of a secondValue 
    with open('testcase/test1.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is False
    # [CASE2]: Add an invalid secondKey
    with open('testcase/test3.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is False
    # [CASE3]: Add an invalid firstKey
    with open('testcase/test4.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is False
    # [CASE4]: Change the data type of a secondValue
    with open('testcase/test8.json', 'r') as test_file:
        inputParams = json.load(test_file)
    assert json_validate(defaultParams, inputParams) is False
