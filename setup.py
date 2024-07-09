from setuptools import setup, find_packages

setup(
    name='MassCalibration',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'PyQt5',
        'toml',
        # Add ROOT installation here if possible
    ],
    entry_points={
        'console_scripts': [
            'MassCalibration=MassCalibration.__main__:main',
        ],
    },
    package_data={
        'MassCalibration': ['config.toml'],
    },
)
