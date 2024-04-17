from setuptools import find_packages
from distutils.core import setup

setup(
    name='rl_air_gym',
    version='1.0.0',
    author='Huyen Dang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='vhdang@mail.uni-padeborn.de',
    description='Reinforcement learning for aerial navigation',
    install_requires=['isaacgym',
                      'matplotlib',
                      'numpy',
                      'torch',
                      'pytorch3d']
)