from setuptools import setup, find_packages

setup(
    name="ab_tools",
    version="0.1.0",
    description="A/B tools",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["hdbscan == 0.8.26","statsmodels == 0.12.2","tqdm == 4.62.3",
        "numpy == 1.20.3", "pandas >= 1.2.4", "pydantic >= 1.8.1", "scipy >= 1.4.1",
        "SQLAlchemy >= 1.1.15", "pytest >= 6.2", "PyYAML >= 5.4.1", "scikit_learn >= 0.24.2", 
        "fastcore ==  1.4.3"
    ],
)
