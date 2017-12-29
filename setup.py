from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sensenet'))
from version import VERSION

setup(name='sensenet',
      version=VERSION,
      include_package_data=True,
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
      keywords = ['deep learning', 'reinforcement learning', 'machine learning','robotics'],
      #package_data={'sensenet': ['data/*']},
      tests_require=['pytest'],
)
