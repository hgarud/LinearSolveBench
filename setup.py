"""Include public runtime assets without maintaining a second source copy."""

from __future__ import annotations

import pathlib
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = pathlib.Path(__file__).resolve().parent
RESOURCE_PATTERNS = (
    "benchmark.toml",
    "data/*.json",
    "data/*.md",
    "data/releases/**/*.json",
    "data/releases/**/*.md",
    "native/CMakeLists.txt",
    "native/include/*.h",
    "native/include/NOTICE-*",
    "native/include/LICENSE-*",
    "native/src/*.cpp",
)


def resource_files() -> list[pathlib.Path]:
    """Allow only release metadata and native sources into installed resources."""
    return sorted(
        {
            path
            for pattern in RESOURCE_PATTERNS
            for path in ROOT.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )


class BuildWithResources(build_py):
    def run(self) -> None:
        super().run()
        destination = pathlib.Path(self.build_lib) / "linear_solver_bench/_resources"
        # A reused build directory must not retain removed release manifests.
        if destination.exists():
            shutil.rmtree(destination)
        for source in resource_files():
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        destination = pathlib.Path(self.build_lib) / "linear_solver_bench/_resources"
        return super().get_outputs(include_bytecode) + [
            str(destination / source.relative_to(ROOT)) for source in resource_files()
        ]


setup(cmdclass={"build_py": BuildWithResources})
