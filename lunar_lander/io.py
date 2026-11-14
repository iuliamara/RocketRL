import csv
import hashlib
import json
import platform
from importlib.metadata import version


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lunar_versions():
    return {"python": platform.python_version(), **{
        package: version(package) for package in ("gymnasium", "Box2D", "stable-baselines3", "torch")
    }}
