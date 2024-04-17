# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from aerial_gym.envs.base.drone_config import DroneCfg
from .base.drone import Drone
import os

from aerial_gym.utils.task_registry import task_registry

task_registry.register("drone", Drone, DroneCfg())
