import glob
import os
import shutil

from invoke import task

import bruces


@task
def build(c):
    shutil.rmtree("dist", ignore_errors=True)
    c.run("python -m pep517.build --source --binary .")


@task
def tag(c):
    c.run("git tag v{}".format(bruces.__version__))
    c.run("git push --tags")


@task
def upload(c):
    c.run("twine upload dist/*")


@task
def clean(c, bytecode=False):
    patterns = [
        "build",
        "dist",
        "bruces.egg-info",
    ]

    if bytecode:
        patterns += glob.glob("**/*.pyc", recursive=True)
        patterns += glob.glob("**/__pycache__", recursive=True)

    for pattern in patterns:
        if os.path.isfile(pattern):
            os.remove(pattern)
        else:
            shutil.rmtree(pattern, ignore_errors=True)


@task
def black(c):
    c.run("black -t py36 bruces")
    c.run("black -t py36 test")


@task
def docstring(c):
    c.run("docformatter -r -i --blank --wrap-summaries 88 --wrap-descriptions 88 --pre-summary-newline bruces")


@task
def isort(c):
    c.run("isort bruces")
    c.run("isort test")


@task
def format(c):
    c.run("invoke isort black docstring")
