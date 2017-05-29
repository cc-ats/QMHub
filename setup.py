from setuptools import setup, find_packages

setup(name='namd_qmmm',
      version='2.12.1',
      description='Custom QM/MM interface for NAMD',
      url='https://github.com/panxl/namd-qmmm',
      author='Xiaoliang Pan',
      author_email='panxl@ou.edu',
      license='MIT',
      packages=find_packages(),
      requires=['numpy (>=1.10.4)'],
      zip_safe=False)
