from setuptools import setup
 
setup(
  name='sensenet',
  version='0.1',
  scripts=['env.py'],
  author='Jason Toy',
  url='http://sensenet.ai',
  install_requires=[ 'numpy>=1.10.4', 'pybullet' ]
)
