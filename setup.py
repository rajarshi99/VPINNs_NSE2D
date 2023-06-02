from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='variational_pinns',
    version='1.0.0',
    packages=find_packages(),
    install_requires=requirements,
)
