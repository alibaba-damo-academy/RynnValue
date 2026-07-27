# adapted from jaxrl2
from setuptools import find_packages, setup

setup(
    name="jaxrl2",
    version="0.1.0",
    packages=[p for p in find_packages() if p == "jaxrl2" or p.startswith("jaxrl2.")],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
    description="jaxrl2 vendored for openpi IQL training.",
)
