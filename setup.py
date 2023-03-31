import re
import sys
import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

def parse_requirements(filename):
    with open(os.path.join(here, filename)) as f:
        reqs = [
            line.strip(' -')
            for line in f.read().splitlines()
            if not any([line.strip(' -').startswith(i) for i in [
                'name', 'dependencies', 'pip',
            ]])
        ]
    return reqs


setup(
    name='zephyr',
    version='1.1.0',
    description='Zero-emissions Electricity system Planning with HourlY operational Resolution',
    author='Patrick Brown',
    author_email='prbrown@mit.edu',
    url='https://github.com/patrickbrown4/zephyr',
    license='MIT',
    packages=['zephyr'],
    install_requires=parse_requirements('environment.yml'),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)
