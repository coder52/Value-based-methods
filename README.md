[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
[image3]: /img/add_interpreter.jpg "New interpreter"
[image4]: /img/conda_py36_interpreter.jpg "Conda interpreter"
[image5]: /img/start_notebook.jpg "start notebook"
[image6]: /img/run_all.jpg "run all"

# Value-Based Methods

![Trained Agents][image1]

This repository contains material related to Udacity's Value-based Methods course.

## Table of Contents

### Tutorials

The tutorials lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

* [Deep Q-Network](https://github.com/coder52/Value-based-methods/tree/main/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.

### Labs / Projects

The labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

* [Navigation](https://github.com/coder52/Value-based-methods/tree/main/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.

### Resources

* [Cheatsheet](https://github.com/coder52/Value-based-methods/tree/main/cheatsheet): You are encouraged to use [this PDF file](https://github.com/udacity/Value-based-methods/blob/main/cheatsheet/cheatsheet.pdf) to guide your study of reinforcement learning. 

## OpenAI Gym Benchmarks

### Box2d
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/coder52/Value-based-methods/blob/main/dqn/exercise/Deep_Q_Network.ipynb) | solved in 824 episodes

## Unity ML Benchmarks

### Banana
- `Banana Collector` with [Deep Q-Networks (DQN)](https://github.com/coder52/Value-based-methods/blob/main/dqn/exercise/Deep_Q_Network.ipynb) | solved in 741 episodes
- `Banana Collector` with [dueling Deep Q-Networks (dueling DQN)](https://github.com/coder52/Value-based-methods/blob/main/p1_navigation/Navigation.ipynb) | solved in 785 episodes

## Instructions for installing dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Download and install [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/?section=windows)

2. Clone the repository using PyCharm's `Project From Version Control` option. 

3. Add new interpreter 

	![New interpreter][image3]

4. Create new conda interpreter with python 3.6 and name `drlnd`

	![Conda interpreter][image4]
    	
5. Open a terminal in PyCharm.  Then, install several dependencies.

````
cd python
pip install torch==1.10.2+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
cd ..
````

## How to run the code, to train and test the agent

1. Start the jupyter notebook in PyCharm-Terminal by writing `jupyter notebook`
	
	![start notebook][image5]

2. Enter the p1_navigation folder from notebook and open `Navigation.ipynb` file
3. Follow the instructions in notebooks for training the agent
4. Or you can click just 

	![run all][image6]

10. To see what trained agents can do, you should run the `run_agent.py` and `run_dueling_agent.py` scripts.

## Want to learn more?

<p align="center">Come learn with us in the <a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">Deep Reinforcement Learning Nanodegree</a> program at Udacity!</p>

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
