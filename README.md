# SenseNet
SenseNet is a sensorimotor and touch simulator to teach AIs how to interact with their environments via sensorimotor systems and touch neurons. SenseNet is meant as a research framework for machine learning researchers and theoretical computational neuroscientists.


![gestures](images/gestures.png?raw=true "gestures")

## Reinforcement learning

 SenseNet can be used in reinforcement learning environments. The original code used OpenAI's gym as the base and so any code written for gym can be used with little to no tweaking of your code. Oftentimes you can just replace gym with sensenet and everything will work. Additionally, SenseNet can be used


## Supported Systems
We currently support Mac OS X and Linux (ubuntu 14.04), Windows mostly works, but we don't have a windows developer.  We also have docker and vagrant/virtualbox images for you to run an any platform that supports them.

## Install from source
git clone http://github.com/jtoy/sensenet
you can run "pip install -r requirements.txt" to install all the python software dependencies
pip install -e '.[all]'


### Install the fast way:
pip install sensenet

##  Train an basic RL agent to learn to touch a missile with "6th sense":
python examples/agents/reinforce.py -e TouchWandEnv-v0




## Dataset

I have made and collected thousands of different objects to manipulate in the simulator.
You can use the SenseNet dataset or your own dataset.
Download the full dataset at https://sensenet.ai

![dataset](images/touchnetv2.png?raw=true "dataset")



## Testing

we use pytest to run tests, to tun the tests just type "cd tests && pytest" from the root directory

## running benchmarks

Included with SenseNet are several examples for competing on the benchmark "blind object classification"
There is a pytorch example and a tensorflow example. to run them:
cd agents && python reinforce.py

to see the graphs: tensorboard --logdir runs then go to your browser at http://localhost:6000/
python setup.py register sdist upload
