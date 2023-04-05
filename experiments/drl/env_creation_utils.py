
import gym
import numpy as np

from envs.mujoco_bots.custom_half_cheetah import CustomHalfCheetahEnv
from gym.wrappers.time_limit import TimeLimit

from envs.mujoco_maze.maze_task import RegularCorridors4Rooms, UnfairCorridors4Rooms, UMaze, UnfairUMaze
from envs.mujoco_maze.maze_task import DoubleCorridorMaze, UnfairDoubleCorridorMaze
from envs.mujoco_maze.maze_env import MazeEnv
from envs.mujoco_bots.point import PointEnv
from envs.mujoco_bots.custom_point import CustomPointEnv

from envs.econ.economy import Economy

SETTING_DICT = {
    # "hc-friction": ["HalfCheetah-v3", "HcF-10"],
    "hc-friction": ["HC-friction-10", "HalfCheetah-v3"],
    "hc-extra-gravity": ["HC-high-gravity", "HalfCheetah-v3"],
    "hc-less-gravity": ["HC-low-gravity", "HalfCheetah-v3"],
    "hc-big-feet": ["HC-big-feet", "HalfCheetah-v3"],
    "hc-small-feet": ["HC-small-feet", "HalfCheetah-v3"],
    "maze": ["Big-Point-4rooms" , "Point-4rooms"],
    "u-maze": ["Big-Point-Umaze" , "Point-Umaze"],
    "corridor": ["Big-Point-Corridor" , "Point-Corridor"],
    "econs-low":["econ-1", "econ-2"],
    "econs-high":["econ-3", "econ-4"],
    "cheetah-family" : ["HC-big-feet", "HC-friction-10", "HalfCheetah-v3"],
}

def get_envs_ids(setting_name:str):
    """

    :param env_setting_name:
    :return:
    """
    assert setting_name in SETTING_DICT.keys(), "Unknown environment group setting!"

    return SETTING_DICT[setting_name]


def get_indv_env(gym_id):
    """
    Wrapper for creating custom and gym environments
    :param gym_id:
    :return:
    """
    # create the corresponding environment here:
    if gym_id == "HC-friction-10":
        env = CustomHalfCheetahEnv(xml_file="ten_fric_half_cheetah.xml")
        env = TimeLimit(env, max_episode_steps=1000)
    elif gym_id == "HC-high-gravity":
        env = CustomHalfCheetahEnv(xml_file="huge_gravity_half_cheetah.xml")
        env = TimeLimit(env, max_episode_steps=1000)
    elif gym_id == "HC-low-gravity":
        env = CustomHalfCheetahEnv(xml_file="small_gravity_half_cheetah.xml")
        env = TimeLimit(env, max_episode_steps=1000)
    elif gym_id == "HC-big-feet":
        env = CustomHalfCheetahEnv(xml_file="big_foot_half_cheetah.xml")
        env = TimeLimit(env, max_episode_steps=1000)
    elif gym_id == "HC-small-feet":
        env = CustomHalfCheetahEnv(xml_file="small_foot_half_cheetah.xml")
        env = TimeLimit(env, max_episode_steps=1000)
    elif gym_id == "Point-4rooms":
        env = MazeEnv(model_cls=PointEnv, maze_task=RegularCorridors4Rooms,
                      maze_size_scaling=RegularCorridors4Rooms.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=RegularCorridors4Rooms.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=100)
    elif gym_id == "Big-Point-4rooms":
        env = MazeEnv(model_cls=CustomPointEnv, maze_task=UnfairCorridors4Rooms,
                      maze_size_scaling=UnfairCorridors4Rooms.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=UnfairCorridors4Rooms.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=100)
        # Note: radius of CustomPoint is 2.5, and the scale of a block in maze is set 4
        #   -> a custom point should be unable to go through just a gap of one empty block
    elif gym_id == "Point-Umaze":
        env = MazeEnv(model_cls=PointEnv, maze_task=UMaze,
                      maze_size_scaling=UMaze.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=UMaze.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=500)
    elif gym_id == "Big-Point-Umaze":
        env = MazeEnv(model_cls=CustomPointEnv, maze_task=UnfairUMaze,
                      maze_size_scaling=UnfairUMaze.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=UnfairUMaze.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=500)
    elif gym_id == "Point-Corridor":
        env = MazeEnv(model_cls=PointEnv, maze_task=DoubleCorridorMaze,
                      maze_size_scaling=DoubleCorridorMaze.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=DoubleCorridorMaze.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=500)
    elif gym_id == "Big-Point-Corridor":
        env = MazeEnv(model_cls=CustomPointEnv, maze_task=DoubleCorridorMaze,
                      maze_size_scaling=DoubleCorridorMaze.MAZE_SIZE_SCALING.point,
                      inner_reward_scaling=DoubleCorridorMaze.INNER_REWARD_SCALING, )
        env = TimeLimit(env, max_episode_steps=500)
    elif gym_id == "econ-1":
        # env = Economy(k=1.0, alpha=0.1, g=0.5, low_dg=-0.01, high_dg=0.0, limit_g_low=0.0)
        env = Economy(k=1.0, alpha=0.25, g=0.25, low_dg=0.0, high_dg=0.5)
        env = TimeLimit(env, max_episode_steps=100)
    elif gym_id == "econ-2":
        # env = Economy(k=1.0, alpha=0.1, g=0.5, low_dg=-0.01, high_dg=0.0, limit_g_low=0.1)
        env = Economy(k=1.0, alpha=0.25, g=0.05, low_dg=0.0, high_dg=0.5)
        env = TimeLimit(env, max_episode_steps=100)
    elif gym_id == "econ-3":
        env = Economy(k=1.0, alpha=0.1, g=0.5, low_dg=-0.01, high_dg=0.0, limit_g_low=0.2)
        env = TimeLimit(env, max_episode_steps=100)
    elif gym_id == "econ-4":
        env = Economy(k=1.0, alpha=0.1, g=0.5, low_dg=-0.005, high_dg=0.0, limit_g_low=0.15)
        env = TimeLimit(env, max_episode_steps=100)
    else:
        env = gym.make(gym_id)

    return env


def make_indv_env(gym_id, seed, idx, capture_video, run_name, log_path, subgroup):
    def thunk():
        env = get_indv_env(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"{log_path}/videos/{run_name}/subgroup_{subgroup}/")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk