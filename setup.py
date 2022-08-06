from setuptools import setup
from mypythontools_cicd import packages

if __name__ == "__main__":
    extras_requirements = {
        i: packages.get_requirements(f"requirements/extras_{i}.txt", path="requirements")
        for i in ["all", "datasets", "datatypes"]
    }

    setup(
        **packages.get_package_setup_args("mydatapreprocessing", development_status="alpha"),
        **packages.personal_setup_args_preset,
        description="Library/framework for making predictions.",
        long_description=packages.get_readme(),
        install_requires=packages.get_requirements("requirements/requirements.txt"),
        extras_require=extras_requirements,
    )
