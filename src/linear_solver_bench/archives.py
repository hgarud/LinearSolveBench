"""Read bounded ZIP members without extracting archive paths."""

from collections.abc import Mapping
from zipfile import ZipFile


def zip_members(archive: ZipFile) -> set[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive members")
    return set(names)


def read_zip_member(archive: ZipFile, name: str, limit: int) -> bytes:
    member = archive.getinfo(name)
    if member.is_dir() or member.file_size > limit:
        raise ValueError(f"archive member {name!r} exceeds its limit or is a directory")
    return archive.read(member)


def read_zip_files(archive: ZipFile, limits: Mapping[str, int]) -> dict[str, bytes]:
    names = zip_members(archive)
    if names - limits.keys():
        raise ValueError("unexpected archive members")
    return {name: read_zip_member(archive, name, limits[name]) for name in names}
