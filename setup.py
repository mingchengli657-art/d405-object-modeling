"""Legacy setuptools entry point for systems with older setuptools.

The canonical metadata lives in pyproject.toml.  This small shim keeps
editable installation working offline on Ubuntu images that still ship
setuptools 59.x.
"""

from setuptools import find_packages, setup


setup(
    name="object-modeling",
    version="0.1.0",
    description="Portable D405 and ChArUco RGB-D object modeling toolkit",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=["numpy>=1.21,<2", "scipy>=1.8,<2", "PyYAML>=6,<7", "opencv-contrib-python>=4.5.4,<4.6"],
    extras_require={"test": ["pytest>=7,<9"]},
    entry_points={"console_scripts": ["object-modeling=object_modeling.cli:main"]},
)
