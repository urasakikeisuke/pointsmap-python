# from skbuild import setup
# from skbuild.command.build_ext import build_ext
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

setup(
    name='pointsmap',
    version='1.0.0',
    description='`https://github.com/shikishima-TasakiLab/pointsmap-python`',
    long_description='`https://github.com/shikishima-TasakiLab/pointsmap-python`',
    author='Junya Shikishima',
    author_email='160442065@ccalumni.meijo-u.ac.jp',
    url='https://github.com/shikishima-TasakiLab',
    license='',
    packages=['pointsmap'],
    ext_modules=[
        Pybind11Extension(
            "pointsmap",
            []
        )
    ]
)
