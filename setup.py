from setuptools import setup, find_packages


setup(
    name='formula1',
    keywords='',
    version='0.1',
    author='Niels Hoogeveen2',
    packages=find_packages(exclude=['data', 'notebooks']),
    entry_points={
        'console_scripts': [
            'run-model=formula1.models.models_utils_GS:run',
            'run-model-2=formula1.models.models_utils_GS_groupkfold:run',
            'run-model-3=formula1.models.models_utils_randomsearch_groupkfold:run'
        ]
    }
)


