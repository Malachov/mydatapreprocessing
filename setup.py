#%%
from setuptools import setup, find_packages
import pkg_resources

if __name__ == "__main__":

    with open("README.md") as readme_file:
        readme = readme_file.read()

    with open("requirements_extras.txt") as f:
        requirements = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

    with open("requirements.txt") as f:
        requirements_extras = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

    with open("mydatapreprocessing/__init__.py") as version_file:
        version = re.findall('__version__ = "(.*)"', version_file.read())[0]

    setup(
        author_email="malachovd@seznam.cz",
        author="Daniel Malachov",
        description="Library/framework for making predictions.",
        include_package_data=True,
        install_requires=requirements,
        requirements_extras={"extras": requirements_extras},
        license="mit",
        long_description_content_type="text/markdown",
        long_description=readme,
        name="mydatapreprocessing",
        packages=find_packages(exclude=("tests**",)),
        platforms="any",
        project_urls={
            "Documentation": "https://mydatapreprocessing.readthedocs.io/",
            "Home": "https://github.com/Malachov/mydatapreprocessing",
        },
        python_requires=">=3.7",
        url="https://github.com/Malachov/mydatapreprocessing",
        version=version,
        classifiers=[
            "Programming Language :: Python",
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Natural Language :: English",
            "Environment :: Other Environment",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Software Development :: Libraries :: Application Frameworks",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
        ],
    )
