from setuptools import setup

setup(
    name='PFFRA',
    version='1.0.1',
    author='Jianqiao Mao',
    author_email='jxm1417@student.bham.ac.uk',
    license='GNU GPL-2.0',
    description="An Interpretable Machine Learning technique to analyse the contribution of features in the frequency domain. This method is inspired by permutation feature importance analysis but aims to quantify and analyse the time-series predictive model's mechanism from a global perspective.",
    url='https://github.com/JianqiaoMao/PFFRA',
    #packages=['PFFRA'],
    install_requires=['numpy', 'matplotlib', 'pywt'],
)
