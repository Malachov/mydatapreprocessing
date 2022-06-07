"""Push the CI pipeline. Format, create commit from all the changes, push and deploy to PyPi."""

from mypythontools_cicd.project_utils import project_utils_pipeline, default_pipeline_config

default_pipeline_config.deploy = True
# default_pipeline_config.do_only = ""


if __name__ == "__main__":

    # default_pipeline_config.do_only = "deploy"

    # All the parameters can be overwritten via CLI args
    project_utils_pipeline(config=default_pipeline_config)
"""Push the CI pipeline. Format, create commit from all the changes, push and deploy to PyPi."""

from mypythontools_cicd.project_utils import project_utils_pipeline, default_pipeline_config

default_pipeline_config.deploy = True
# default_pipeline_config.do_only = ""


if __name__ == "__main__":

    # default_pipeline_config.do_only = "deploy"

    # All the parameters can be overwritten via CLI args
    project_utils_pipeline(config=default_pipeline_config)
