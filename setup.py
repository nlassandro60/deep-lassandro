from setuptools import setup, find_packages

setup(
    name="spare",
    version="0.1.0",
    author="Yu Zhao",
    author_email="yu.zhao@ed.ac.uk",
    description=r'Code for the paper "Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering"',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yuzhaouoe/SAE-based-representation-engineering",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
