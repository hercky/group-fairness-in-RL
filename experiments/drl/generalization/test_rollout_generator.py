"""
Cite cleanrl and focops here:


"""
import numpy as np
import torch



def sample_return(agent, env, device):
    """
    :return:
    """
    eps_return = None

    # reset all environments
    next_obs = torch.Tensor(env.reset()).to(device)

    done = False

    while not done:
        with torch.no_grad():

            action, _, _, _ = agent.get_action_and_value(next_obs.view(1, -1))

            next_obs, reward, done, info = env.step(action.view(-1,).cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)

            # store the average statistics for this particular global step
            if "episode" in info.keys():
                eps_return = info['episode']['r']
                return eps_return

    return eps_return
