from setuptools import find_packages, setup
__version__ = '0.0.1' 
DESCRIPTION = 'QOSF MNIST QML Package'
LONG_DESCRIPTION = 'QOSF Mentorship program opensource program'

setup(
    name="qosf",
    version=__version__,
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.7",
    project_urls={
        "Documentation": "https://github.com/rryoung98/qml",
        "Source Code": "https://github.com/rryoung98/qml",
        "Tutorials": "https://github.com/rryoung98/qml/tree/main/demos",
    }
)
