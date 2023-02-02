import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# import ``__version__`` from code base
exec(open('cellbox/version.py').read())

setuptools.setup(
    name="cellbox",
    version=__version__,
    author="Bo Yuan, Judy Shen, Augustin Luna",
    author_email="boyuan@g.harvard.edu, c_shen@g.harvard.edu, augustin_luna@hms.harvard.edu",
    description="A mechine learning framework with a mathematical core of differential equations to train network models on perturbation-response data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dfci/CellBox",
    packages=['cellbox'],
    python_requires='>=3.6',
    install_requires=['tensorflow==2.6.2', 'numpy==1.19.5', 'pandas==0.24.2', 'scipy==1.3.0'],
    tests_require=['pytest', 'pandas', 'numpy', 'scipy'],
    setup_requires=['pytest-runner', "pytest"],
    zip_safe=True,
    keywords='Scientific machine learning, sciML, perturbation biology, interpretability',
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
