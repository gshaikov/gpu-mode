from setuptools import setup
from setuptools.command.build_ext import build_ext
import subprocess
import os


class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../cpp/build")
        )
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"], cwd=build_dir
        )
        super().run()


setup(
    name="matrix_metal",
    version="0.1.0",
    description="Python library for Metal-accelerated matrix multiplication",
    author="Your Name",
    packages=["matrix_metal"],
    install_requires=["numpy"],
    cmdclass={"build_ext": CMakeBuild},
)
