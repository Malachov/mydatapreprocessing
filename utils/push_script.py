import mypythontools


if __name__ == "__main__":
    # mypythontools.utils.push_pipeline(
    #     deploy=True,
    #     test_options={"requirements": ["requirements.txt", "requirements_advanced.txt"], "verbose": True},
    # )  # , test_options={"use_virutalenv": False}
    mypythontools.utils.push_pipeline(
        deploy=True, test_options={"use_virutalenv": False}
    )
