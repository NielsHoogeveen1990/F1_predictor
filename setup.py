from setuptools import setup, find_packages


setup(
    name='formula1',
    keywords='',
    version='0.1',
    author='Pim Hoeven',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            # name you put in bash = function that you call
            'formula1 = formula1.main:main'
        ]
    }
)

