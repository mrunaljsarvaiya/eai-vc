#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, PlannerGoal, Collisions

from habitat_vc.rl.measures import AngleSuccess, AngleToGoal, TrainSuccess


@registry.register_measure
class SimpleReward(Measure):
    cls_uuid: str = "simple_reward"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        super().__init__(**kwargs)
        self._sim = sim
        self._config = config
        self._previous_dtg: Optional[float] = None
        self._previous_atg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToGoal.cls_uuid,
                Collisions.cls_uuid,
                PlannerGoal.cls_uuid,
                TrainSuccess.cls_uuid,
                AngleToGoal.cls_uuid,
                AngleSuccess.cls_uuid,
            ],
        )
        self._metric = None
        self._previous_dtg = None
        self._previous_sdtg = None
        self._previous_atg = None
        self.update_metric(task=task)


    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        
        # success
        success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        success_reward = self._config.SUCCESS_REWARD if success else 0.0

        # distance-to-goal
        dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        if self._previous_dtg is None:
            self._previous_dtg = dtg
        add_dtg = self._config.USE_DTG_REWARD
        dtg_reward = self._previous_dtg - dtg if add_dtg else 0.0
        self._previous_dtg = dtg

        # Shortest path 
        sdtg = task.measurements.measures[PlannerGoal.cls_uuid].get_metric()
        sdtg = self._config.SDTG_REWARD_SCALE * sdtg

        # Collision cost
        collision_penalty = 0.0
        # scale = 0.5
        # if task.measurements.measures[Collisions.cls_uuid].get_metric() is not None:
        #     collision = task.measurements.measures[Collisions.cls_uuid].get_metric()["is_collision"]

        #     if collision:
        #         collision_penalty = -1.0 * scale

        # angle-to-goal
        atg = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()
        add_atg = self._config.USE_ATG_REWARD
        if self._config.USE_ATG_FIX:
            if dtg > self._config.ATG_REWARD_DISTANCE:
                atg = np.pi
        else:
            if dtg > self._config.ATG_REWARD_DISTANCE:
                add_atg = False
        if self._previous_atg is None:
            self._previous_atg = atg
        atg_reward = self._previous_atg - atg if add_atg else 0.0
        self._previous_atg = atg

        # angle success
        angle_success = task.measurements.measures[AngleSuccess.cls_uuid].get_metric()
        angle_success_reward = (
            self._config.ANGLE_SUCCESS_REWARD if angle_success else 0.0
        )

        # action = task.last_action

        # collision_penalty = 0.0
        # if hasattr(task, "last_observation") and hasattr(task, "last_action"):
            
        #     if task.last_observation is not None:

        #         local_map = task.last_observation["local_top_down_map"]
        #         local_map = (local_map > 0).astype(np.uint8)
        #         center = [local_map.shape[0] // 2, local_map.shape[1] // 2]
        #         agent_state = task._sim.get_agent_state()
        #         heading = np.rad2deg(agent_state.rotation.euler_angles[1])
        #         action = int(task.last_action)

        #         if self.will_collide(local_map, action, heading, center):
        #             print("Collision")
        #             collision_penalty = -1.0

        # slack penalty
        slack_penalty = self._config.SLACK_PENALTY


        self._metric = (
            success_reward
            + dtg_reward
            + atg_reward
            + angle_success_reward
            + slack_penalty 
            + collision_penalty
        )
