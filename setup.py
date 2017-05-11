from setuptools import setup, find_packages

version = '0.0.1'

setup(name='dense-tensor',
      version=version,
      description='Neural tensor layer for Keras',
      url='https://github.com/bstriner/dense_tensor',
      author='Ben Striner',
      author_email='bstriner@gmail.com',
      packages=find_packages(),
      install_requires=['Keras'],
      keywords=['keras', 'tensor', 'neural tensor network'],
      license='MIT',
      classifiers=[
          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'
      ])
