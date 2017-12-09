# SenseNet
SenseNet is a sensorimotor simulator and dataset of touchable 3D objects to teach AIs how to interact with their environments via sensorimotor systems and touch neurons. SenseNet is meant as a research framework for machine learning researchers and theoretical neuroscientists. 

# Goals

* standardized platform to reduce development time for researchers, focus on theory, not physics tweaking
* a suite benchmarks to compare and contrast against
* easily test new sensorimotor and control theories


![gestures](images/gestures.png?raw=true "gestures")

## Supported Systems
We currently support Mac OS X and Linux (ubuntu 14.04), Windows mostly works, but we don't have a windows developer.  We also have vagrant/virtualbox images for you to run an any platform that supports virtualbox

## Installation
you will need python3 (python2 might work, but has not been tested), numpy, and pybullet

### Install the fast way:
pip install sensenet (not done)

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

by default the dataset is expected to be in this structure:

dir/sensenet_data/objects/0
main_dir/sensenet_data/objects/1
dir/sensenet_data/objects/2

![dataset](images/dataset.png?raw=true "dataset")



## Testing

we use pytest to run tests, ro tun the tests just type "cd tests && pytest" from the root directory

## running benchmarks
 
Included with SenseNet are several examples for competing on the benchmark "blind object classification"
There is a pytorch example and a tensorflow example. to run them:
cd agents && python reinforce.py

to see the graphs: tensorboard --logdir runs then go to your browser at http://localhost:6000/
