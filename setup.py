"""Python setup.py for tsqa_pipeline_v2 package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("tsqa_pipeline_v2", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="tsqa_pipeline_v2",
    version=read("tsqa_pipeline_v2", "VERSION"),
    description="Awesome tsqa_pipeline_v2 created by mauricekraus",
    url="https://github.com/mauricekraus/tsqa-pipeline-v2/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="mauricekraus",
    packages=find_packages(exclude=[".docker", ".devcontainer", ".github"]),
    install_requires=read_requirements(
        ["requirements.txt", "lightning-requirements.txt"]
    ),
    entry_points={"console_scripts": ["tsqa_pipeline_v2 = tsqa_pipeline_v2.__main__:main"]},
)
