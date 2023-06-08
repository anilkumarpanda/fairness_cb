from setuptools import find_packages, setup

setup(
    name="faircb",
    version="0.1",
    author="Anil Panda",
    author_email="anilkumar.panda@ing.com",
    description="Python package code breakfast in Fairness",
    long_description="Python package code breakfast in Fairness",
    url="https://github.com/anilkumarpanda/fairness_cb",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
