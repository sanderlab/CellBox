import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# import ``__version__`` from code base
exec(open('pertbio/version.py').read())

setuptools.setup(
    name="pertbio",
    version=__version__,
    author="Bo Yuan, Judy Shen, Augustin Luna",
    author_email="boyuan@g.harvard.edu, c_shen@g.harvard.edu, augustin_luna@hms.harvard.edu",
    description="A mechine learning framework with a mathematical core of differential equations to train network models on perturbation-response data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dfci/CellBox",
    packages=['pertbio'],
    python_requires='>=3.6',
    install_requires=['tensorflow==1.15.0', 'numpy==1.16.0', 'pandas'],
    tests_require=['pytest', 'pandas'],
    setup_requires=['pytest-runner', "pytest"],
    zip_safe=True,
    keywords='Machine learning, perturbation biology',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: LGPL",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

)
