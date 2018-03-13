from setuptools import setup


setup(name='stdsct',
      version='1.0',
      description='Edge-Averaged Finite Elements (EAFE) for FENiCS',
      url='https://github.com/arthbous/StochasticDescent',
      author='Arthur Bousquet, Qipin Chen, Shuonan Wu',
      author_email='',
      license='GNU GENERAL PUBLIC LICENSE',
      install_requires=['numpy','scipy'],
      packages=['stdsct'],
      zip_safe=False)
