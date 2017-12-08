from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sensenet'))
from version import VERSION

setup(name='sensenet',
      version=VERSION,
      description='SenseNet',
      url='https://github.com/jtoy/sensenet',
      author='Jason Toy',
      author_email='support@sensenet.ai',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('sensenet')],
      zip_safe=False,
      install_requires=[
          'numpy>=1.10.4', 'requests>=2.0', 'six', 'pybullet'
      ],
      package_data={'sensenet': ['data/*']},
      tests_require=['pytest'],
)
