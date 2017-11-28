# SenseNet
SenseNet is a dataset of touchable 3D objects and a sensorimotor simulator to teach AIs how to interact with their environments via sensorimotor systems and touch neurons. SenseNet is meant as a research tool for computational neuroscientists and machine learning researchers. 

You can easily run your own experiments with our environments and agents or you can build your own

![gestures](images/gestures.png?raw=true "gestures")
for more information, visit http://sensenet.ai

## Supported Systems
We currently support Mac OS X and Linux (ubuntu 14.04). Windows should work, but we have not tested it.  We also have vagrant/virtualbox images for you to run an any platform that supports virtualbox

## Installation
you will need python3 (python2 might work, but has not been tested), numpy, and pybullet

### Install the fast way:
pip install sensenet (still working on this)

## Install from source
git clone http://github.com/jtoy/sensenet
you can run "pip install -r requirements.txt" to install all the python software dependencies
pip install -e '.[all]'

## Virtual images
You can run all the code in a vagrant/virtualbox.  To use the image, install vagrant and virtualbox.  Then run:

vagrant up
once inside the image, you can run "cd /vagrant && python3 test.py"

## dataset
You can use the SenseNet dataset or your own dataset.
Download the dataset from http://sensenet.ai
by default the dataset is expected to be in this structure:

sensenet_main_dir
sensenet_main_dir/sensenet/ #git cloned here
sensenet_main_dir/sensenet_data/ #dataset here
sensenet_main_dir/sensenet_data/objects
sensenet_main_dir/sensenet_data/objects/0
sensenet_main_dir/sensenet_data/objects/1
sensenet_main_dir/sensenet_data/objects/2

![dataset](images/dataset.png?raw=true "dataset")

#the numbers in objects represent the class number for the objects


##Testing
we use pytest to run tests, ro tun the tests just type "pytest" from the root directory

## running benchmarks
 
Included with SenseNet are several examples for competing on the benchmark "blind object classification"
There is a pytorch example and a tensorflow example. to run them:
cd agents && python reinforce.py

to see the graphs: tensorboard --logdir runs then go to your browser at http://localhost:6000/
