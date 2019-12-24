/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.submarine.commons.runtime.param;

import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.commons.runtime.Framework;

import java.util.List;

public interface Parameter {
  /**
   * Get the ML framework
   * @return Framework
   */
  Framework getFramework();

  Parameter setFramework(Framework framework);

  BaseParameters getParameters();

  Parameter setParameters(BaseParameters parameters);

  String getOptionValue(String option) throws YarnException;

  boolean hasOption(String option);

  List<String> getOptionValues(String option) throws YarnException;
}
