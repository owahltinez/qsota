import setuptools

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
  requirements = f.read().splitlines()

setuptools.setup(
    name="qsota",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    package_data={"qsota": ["*.txt"]},
    entry_points={
        "console_scripts": [
            "qsota = qsota.qsota:cli",
        ],
    },
)
