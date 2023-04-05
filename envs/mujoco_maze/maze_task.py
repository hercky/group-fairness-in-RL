"""
Implements a minimalistic version of the MazeTask class (without blocks, billiards, spin, etc..)

Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np

from envs.mujoco_maze.maze_env_utils import MazeCell




class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    """
    [h] Class for defining the goals:

        pos: goal position
        reward_scale: reward on reaching goal
        rgb: color of cell ?
        threshold: for deciding if any position is close to the goal or not
        custom_size: ?
    """
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 0.6,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float:
        """
        returns if the obs is close to the goal or not based

        :param obs:
        :return:
        """
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        """
        returns the euclidean distance to the goal
        :param obs:
        :return:
        """
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]



class MazeTask(ABC):
    """
    [h] Defines the task description

    Attributes:
        REWARD_THRESHOLD:
        PENALTY:
        MAZE_SIZE_SCALING:
        INNER_REWARD_SCALING:
    """
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(8.0, 4.0, 4.0)
    INNER_REWARD_SCALING: float = 0.01

    # dummy arguments n
    # For Fall/Push/BlockMaze
    OBSERVE_BLOCKS: bool = False
    # For Billiard
    OBSERVE_BALLS: bool = False
    OBJECT_BALL_SIZE: float = 1.0
    # Unused now
    PUT_SPIN_NEAR_AGENT: bool = False
    TOP_DOWN_VIEW: bool = False

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass




class Unfair4Rooms(MazeTask):
    """
    Based on the 4Rooms environment
        * In the original maze_task, it builds on the GoalReward4Rooms

    Point4Rooms-v2:
        - Point: the kind of agent (Point env)
        - v2: Uses the SubGoal4Rooms
    """
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    # MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, B, E, E, E, B],
            [B, B, E, B, B, B, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]



class UnfairSubGoal4Rooms(Unfair4Rooms):
    """
    v2 version, add a small reward in other two rooms
    """
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals += [
            MazeGoal(np.array([0.0 * scale, -6.0 * scale]), 0.5, GREEN),
            MazeGoal(np.array([6.0 * scale, 0.0 * scale]), 0.5, GREEN),
        ]




class RegularCorridors4Rooms(MazeTask):
    """
    Based on the 4Rooms environment
        * In the original maze_task, it builds on the GoalReward4Rooms

    Point4Rooms-v2:
        - Point: the kind of agent (Point env)
        - v2: Uses the SubGoal4Rooms
    """
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    # MAZE_SIZE_SCALING: Scaling = Scaling(4.0, 4.0, 4.0)
    # MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=2.0, swimmer=4.0)
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        # self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]
        self.goals = [MazeGoal(np.array([6.0 * scale, 0.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, B, E, E, B, E, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]


class UnfairCorridors4Rooms(MazeTask):
    """
    Based on the 4Rooms environment
        * In the original maze_task, it builds on the GoalReward4Rooms

    Point4Rooms-v2:
        - Point: the kind of agent (Point env)
        - v2: Uses the SubGoal4Rooms
    """
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=4.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        # self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]
        self.goals = [MazeGoal(np.array([6.0 * scale, 0.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R, V = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.PHANTOM_BLOCK
        return [
            [B, B, B, B, B, B, B, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, B],
            [B, B, E, E, B, E, E, B, B],
            [B, E, E, E, B, E, E, E, B],
            [B, E, E, E, V, E, E, E, B],
            [B, R, E, E, B, E, E, E, B],
            [B, B, B, B, B, B, B, B, B],
        ]
#
# class UMaze(MazeTask):
#     REWARD_THRESHOLD: float = 0.9
#     # PENALTY: float = -0.0001
#     PENALTY: float = -0.05
#     # default point scaling is 4
#
#     def __init__(self, scale: float) -> None:
#         super().__init__(scale)
#         self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]
#
#     def reward(self, obs: np.ndarray) -> float:
#         return 1.0 if self.termination(obs) else self.PENALTY
#
#     @staticmethod
#     def create_maze() -> List[List[MazeCell]]:
#         E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
#         return [
#             [B, B, B, B, B, B, B, B, B],
#             [B, R, E, E, E, E, E, E, B],
#             [B, E, B, B, B, B, E, E, B],
#             [B, E, E, E, E, E, E, E, B],
#             [B, B, B, B, B, B, B, B, B],
#         ]
#
#
#
# class UnfairUMaze(MazeTask):
#     REWARD_THRESHOLD: float = 0.9
#     # PENALTY: float = -0.0001
#     PENALTY: float = -0.05
#
#     def __init__(self, scale: float) -> None:
#         super().__init__(scale)
#         self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]
#
#     def reward(self, obs: np.ndarray) -> float:
#         return 1.0 if self.termination(obs) else self.PENALTY
#
#     @staticmethod
#     def create_maze() -> List[List[MazeCell]]:
#         E, B, R, V = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.PHANTOM_BLOCK
#         return [
#             [B, B, B, B, B, B, B, B, B],
#             [B, R, E, E, E, E, E, E, B],
#             [B, V, B, B, B, B, E, E, B],
#             [B, E, E, E, E, E, E, E, B],
#             [B, B, B, B, B, B, B, B, B],
#         ]



class UMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    # PENALTY: float = -0.0001
    PENALTY: float = -0.05
    # default point scaling is 4

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, R, E, E, E, E, B],
            [B, E, B, B, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]



class UnfairUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    # PENALTY: float = -0.0001
    PENALTY: float = -0.05

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R, V = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.PHANTOM_BLOCK
        return [
            [B, B, B, B, B, B, B],
            [B, R, E, E, E, E, B],
            [B, V, B, B, E, E, B],
            [B, E, E, E, E, E, B],
            [B, B, B, B, B, B, B],
        ]




class DoubleCorridorMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    # PENALTY: float = -0.0001
    PENALTY: float = -0.05
    # default point scaling is 4

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B],
            [B, R, E, E, E, E, E, B],
            [B, B, B, E, B, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B],
        ]



class UnfairDoubleCorridorMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    # PENALTY: float = -0.0001
    PENALTY: float = -0.05

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0.0, 2.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R, V = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT, MazeCell.PHANTOM_BLOCK
        return [
            [B, B, B, B, B, B, B, B],
            [B, R, E, E, E, E, E, B],
            [B, B, B, V, B, E, E, B],
            [B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B],
        ]