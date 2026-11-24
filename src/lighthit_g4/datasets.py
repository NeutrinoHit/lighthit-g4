"""Geant4 data sets for the C++ core.

The list is that of Geant4 11.4.1 (``cmake/Modules/G4DatasetDefinitions.cmake``:
name, version, file name, environment variable, MD5).  A data set is taken,
in this order, from its environment variable, from ``LIGHTHIT_G4_DATA``, from
``~/.geant4_pybind`` (the same versions are used by geant4_pybind 0.1.3) or
from ``~/.lighthit-g4/geant4-data``; a missing one is downloaded there from
the Geant4 data server and checked against its MD5.
"""
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

__all__ = ["DATASETS", "ensure_datasets", "data_directory"]

URL = "https://cern.ch/geant4-data/datasets"

# name, version, file name, environment variable, md5
DATASETS = [
    ("G4NDL", "4.7.1", "G4NDL", "G4NEUTRONHPDATA", "54f0ed3995856f02433d42ec96d70bc6"),
    ("G4EMLOW", "8.8", "G4EMLOW", "G4LEDATA", "328330009df633f7e9b3a9f445745298"),
    ("PhotonEvaporation", "6.1.2", "G4PhotonEvaporation", "G4LEVELGAMMADATA",
     "d80ba9bdefcf9e23487a26adfa273304"),
    ("RadioactiveDecay", "6.1.2", "G4RadioactiveDecay", "G4RADIOACTIVEDATA",
     "20d494f73d4bddabd7fab5c06a58895c"),
    ("G4PARTICLEXS", "4.2", "G4PARTICLEXS", "G4PARTICLEXSDATA", "b63341cc97a472b04ff607b094493174"),
    ("G4PII", "1.3", "G4PII", "G4PIIDATA", "05f2471dbcdf1a2b17cbff84e8e83b37"),
    ("RealSurface", "2.2", "G4RealSurface", "G4REALSURFACEDATA", "ea8f1cfa8d8aafd64b71fb30b3e8a6d9"),
    ("G4SAIDDATA", "2.0", "G4SAIDDATA", "G4SAIDXSDATA", "d5d4e9541120c274aeed038c621d39da"),
    ("G4ABLA", "3.3", "G4ABLA", "G4ABLADATA", "b25d093339e1e4532e31038653580ca6"),
    ("G4INCL", "1.3", "G4INCL", "G4INCLDATA", "72c3ff5f7f1e6f727f2ebf3b11ce784f"),
    ("G4ENSDFSTATE", "3.0", "G4ENSDFSTATE", "G4ENSDFSTATEDATA", "c500728534ce3e9fb2fefa0112eb3a74"),
    ("G4CHANNELING", "2.0", "G4CHANNELING", "G4CHANNELINGDATA", "e853fbcba1e8640f7c3a437253955482"),
]


def _candidates():
    if os.environ.get("LIGHTHIT_G4_DATA"):
        yield Path(os.environ["LIGHTHIT_G4_DATA"])
    yield Path.home() / ".geant4_pybind"
    yield Path.home() / ".lighthit-g4" / "geant4-data"


def data_directory():
    """Directory where missing data sets are downloaded."""
    if os.environ.get("LIGHTHIT_G4_DATA"):
        return Path(os.environ["LIGHTHIT_G4_DATA"])
    return Path.home() / ".lighthit-g4" / "geant4-data"


def _download(name, version, filename, md5, target, verbose):
    archive = target / f"{filename}.{version}.tar.gz"
    url = f"{URL}/{archive.name}"
    print(f"lighthit-g4: downloading Geant4 data set {name} {version} from {url}", file=sys.stderr)
    target.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".part")
    digest = hashlib.md5()
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != md5:
        partial.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}")
    partial.rename(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    archive.unlink()


def ensure_datasets(*, download=True, verbose=0):
    """Set the environment variable of every data set, downloading missing ones."""
    for name, version, filename, variable, md5 in DATASETS:
        if os.environ.get(variable) and Path(os.environ[variable]).is_dir():
            continue
        found = next((base / f"{name}{version}" for base in _candidates()
                      if (base / f"{name}{version}").is_dir()), None)
        if found is None:
            if not download:
                raise FileNotFoundError(f"Geant4 data set {name}{version} not found")
            _download(name, version, filename, md5, data_directory(), verbose)
            found = data_directory() / f"{name}{version}"
        os.environ[variable] = str(found)
