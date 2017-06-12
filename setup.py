from distutils.core import setup

setup(
    name='SLP.py',
    version='0.1',
    author='Kenjy Demeester',
    author_email='kenjy.demeester@polymtl.com',
    packages=['SLP.py'],
    scripts=[],
    description='Implementation of sequential linear programming using CyLP.',
    long_description=open('README.txt').read(),
    install_requires=[  "NLP.py",
                        "CyLP",],)
