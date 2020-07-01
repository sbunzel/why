import os
import shutil
import subprocess
from distutils.cmd import Command
from runpy import run_path

from setuptools import find_packages, setup

# read the program version from version.py (without loading the module)
__version__ = run_path("src/why/version.py")["__version__"]


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class DistCommand(Command):

    description = "build the distribution packages (in the 'dist' folder)"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if os.path.exists("build"):
            shutil.rmtree("build")
        subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"])


setup(
    name="why",
    version=__version__,
    author="Steffen Bunzel",
    author_email="steffenbunzel@posteo.de",
    description="An exploration of the world of interpretable machine learning",
    license="MIT",
    url="https://github.com/sbunzel/why",
    packages=find_packages("src"),
    package_dir={"": "src"},
    long_description=read("README.md"),
    install_requires=[],
    tests_require=["pytest", "pytest-cov", "pre-commit"],
    cmdclass={"dist": DistCommand},
    platforms="any",
    python_requires=">=3.7",
    zip_safe=False,
)
