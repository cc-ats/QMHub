from setuptools import setup, find_packages

setup(name='QMHub',
      version='2.12.1',
      description='A universal QM/MM interface.',
      url='https://github.com/cc-ats/QMHub',
      author='Xiaoliang Pan',
      author_email='panxl@ou.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy (>=1.10.4)'],
      zip_safe=False)
