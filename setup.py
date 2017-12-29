from setuptools import setup, find_packages
from codecs import open
from os import path
#from version import VERSION

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(

	name='sensenet',

	version="1.2.0",

	description='SenseNet',

	url='https://github.com/jtoy/sensenet',
	author='Jason Toy',
    author_email='support@sensenet.ai',


	keywords = ['deep learning', 'reinforcement learning',
		 'machine learning','robotics'
	],

	packages=[package for package in find_packages()
                if package.startswith('sensenet')],

    install_requires=[
          'numpy>=1.10.4', 'requests>=2.0', 'six', 'pybullet'
      ],

    tests_require=['pytest']

	)
