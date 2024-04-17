# Copyright (c) 2024, Regelungs- und Automatisierungstechnik (RAT) - Paderborn University
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from rl_gym.envs.base.quad_config import QuadConfig
from .base.quadrotor import Quadrotor
import os

from rl_gym.utils.task_registry import task_registry

task_registry.register("drone", Quadrotor, QuadConfig())
