import torch
import numpy as np
from unityagents import UnityEnvironment
from agent import Agent
from collections import deque


def load_environment(file_name: str, no_graphics: bool = True) -> tuple:
    """Load Unity environment and return brain name and brain object."""
    env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    return env, brain_name, brain


def load_agent(checkpoint_path: str, dueling: bool) -> Agent:
    """Initialize the agent and load pre-trained weights."""
    agent = Agent(state_size=37, action_size=4, seed=0, dueling=dueling)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    return agent


def evaluate_agent(env, brain_name: str, agent: Agent) -> float:
    """Run one episode and return the score."""
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0

    while True:
        action = agent.act(state)
        env_info = env.step([action])[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        score += reward
        state = next_state

        if done:
            break

    return score


def dqn_trainer(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, file_name="checkpoint"):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step([action])[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), file_name + '.pth')
            break

    return scores