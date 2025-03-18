#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import NavigationEpisode
import cv2
from gym import spaces
from habitat.utils.visualizations import maps
import cv2
import time 

# fmt: off
@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoalrotation"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode ** NEW **
        if self.config.SAMPLE_ANGLE == True:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal
    
@registry.register_sensor
class LocalTopDownMapSensor(Sensor):
    """
    Sensor that returns a local top-down occupancy map centered on the agent.
    """

    cls_uuid: str = "local_top_down_map"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self.map_resolution = config.MAP_RESOLUTION  # Full map resolution
        self.local_map_size = config.LOCAL_MAP_SIZE  # Local window size
        self.local_map = None 

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE  # Generic sensor type

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=255, shape=(self.local_map_size, self.local_map_size), dtype=np.uint8)

    def get_observation(
        self, *args: Any, observations, episode: NavigationEpisode, **kwargs: Any
    ):
        t = time.time()        
        # Get agent state (position)
        agent_state = self._sim.get_agent_state()
        agent_x, agent_y, agent_z = agent_state.position  # (x, y, z)

        # Get full top-down map
        if self.local_map is None:
            t1 = time.time()
            height = 2
            full_map = maps.get_topdown_map(
                self._sim.pathfinder, map_resolution=self.map_resolution, height=height
            )
            print(f"get topdown t {time.time() - t1}")
            # Debugging: Print map info
            if full_map is None or full_map.size == 0:
                print("ERROR: Full map is empty! Returning a blank map.")
                import pdb; pdb.set_trace()
                return np.zeros((self.local_map_size, self.local_map_size), dtype=np.uint8)

            self.local_map = full_map 

        # Get map bounds to correctly transform agent position
        (min_bound, _max_bound) = self._sim.pathfinder.get_bounds()

        # Convert world coordinates to pixel coordinates
        map_h, map_w = self.local_map.shape  # Height & Width of map
        grid_x = int(((agent_x - min_bound[0]) / (_max_bound[0] - min_bound[0])) * map_w)
        grid_y = int(((agent_z - min_bound[2]) / (_max_bound[2] - min_bound[2])) * map_h)

        # Extract a local region centered on the agent
        half_size = self.local_map_size // 2
        min_x = max(grid_x - half_size, 0)
        max_x = min(grid_x + half_size, map_w)
        min_y = max(grid_y - half_size, 0)
        max_y = min(grid_y + half_size, map_h)

        if min_x >= max_x or min_y >= max_y:
            print("ERROR: Computed local map bounds are invalid! Returning blank map.")
            import pdb; pdb.set_trace()
            return np.zeros((self.local_map_size, self.local_map_size), dtype=np.uint8)

        local_map = self.local_map[min_y:max_y, min_x:max_x]

        # If the extracted local_map is empty, return a blank map
        if local_map.size == 0:
            print("WARNING: Extracted local map is empty! Returning a blank map.")
            return -1*np.ones((self.local_map_size, self.local_map_size), dtype=np.uint8)

        # Resize the extracted local map to ensure consistent size
        if local_map.shape != (self.local_map_size, self.local_map_size):
            local_map = cv2.resize(local_map, (self.local_map_size, self.local_map_size), interpolation=cv2.INTER_NEAREST)

        # print(f"get obs time {time.time() - t}")
        return local_map
