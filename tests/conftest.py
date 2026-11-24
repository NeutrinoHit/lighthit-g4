"""Samples made once per test run, each in its own process (one Geant4 run manager per process)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def simulate(path, *args):
    env = dict(os.environ)
    # LIGHTHIT_G4_INSTALLED=1 tests the installed package (a wheel) instead of src/
    if not env.get("LIGHTHIT_G4_INSTALLED"):
        env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run([sys.executable, "-m", "lighthit_g4", "-o", str(path), *map(str, args)],
                            env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-3000:] + result.stderr[-3000:])
    return path


TAU_EVENTS = 2000
TAU_ENERGY_GEV = 100.0


@pytest.fixture(scope="session")
def tau_sample(tmp_path_factory):
    """Taus without secondaries: only the tau is tracked, daughters are listed."""
    path = tmp_path_factory.mktemp("tau") / "tau.h5"
    return simulate(path, "--pdg", 15, "--energy-GeV", TAU_ENERGY_GEV, "-n", TAU_EVENTS,
                    "--secondaries", "none", "--tau-decayer", "geant4", "--seed", 7)


PYTHIA_EVENTS = 5000


@pytest.fixture(scope="session", params=[-1.0, 1.0], ids=["left", "right"])
def pythia_sample(request, tmp_path_factory):
    """Taus decayed by Pythia with helicity -1 or +1 (needs pythia8mc)."""
    pytest.importorskip("pythia8mc")
    path = tmp_path_factory.mktemp("pythia") / "tau.h5"
    simulate(path, "--pdg", 15, "--energy-GeV", TAU_ENERGY_GEV, "-n", PYTHIA_EVENTS,
             "--secondaries", "none", "--tau-decayer", "pythia",
             "--polarization", request.param, "--seed", 5)
    return path, request.param


SHOWER_CENTER = (10.0, -20.0, 30.0)
SHOWER_START = (12.0, -21.0, 25.0)


SHOWER_ARGS = ("--pdg", 11, "--energy-GeV", 1.0, "-n", 3, "--seed", 3,
               "--radius-m", 50, "--height-m", 100,
               "--center", *SHOWER_CENTER, "--position", *SHOWER_START,
               "--direction", 0, 0.6, 0.8)


@pytest.fixture(scope="session")
def shower_sample(tmp_path_factory):
    """Full 1 GeV electron showers in a shifted water cylinder (C++ core)."""
    path = tmp_path_factory.mktemp("shower") / "electron.h5"
    return simulate(path, *SHOWER_ARGS)


@pytest.fixture(scope="session")
def pybind_shower_sample(tmp_path_factory):
    """The same showers through geant4_pybind (needs geant4_pybind)."""
    pytest.importorskip("geant4_pybind")
    path = tmp_path_factory.mktemp("pybind") / "electron.h5"
    return simulate(path, *SHOWER_ARGS, "--backend", "geant4_pybind")


GEOMETRY_ARGS = ("--radius-m", 50, "--height-m", 100, "--center", *SHOWER_CENTER)

# two events from one --primaries file: a tau with hadrons at one vertex (a
# stand-in for a nu_tau CC final state) plus a delayed muon elsewhere, and one gamma
MULTI_EVENTS = [
    [{"pdgid": 15, "energy_GeV": 20.0, "position_m": [12.0, -21.0, 25.0], "direction": [0, 0.6, 0.8],
      "polarization": -1.0},
     {"pdgid": 211, "energy_GeV": 3.0, "position_m": [12.0, -21.0, 25.0], "direction": [0.3, 0.5, 0.8]},
     {"pdgid": 2212, "energy_GeV": 0.5, "position_m": [12.0, -21.0, 25.0], "direction": [-0.4, 0.2, 0.9]},
     {"pdgid": 13, "energy_GeV": 1.0, "position_m": [5.0, -15.0, 40.0], "direction": [1, 0, 0],
      "time_ns": 25.0}],
    [{"pdgid": 22, "energy_GeV": 0.7, "position_m": [10.0, -20.0, 30.0], "direction": [0, 0, -1]}],
]


def write_json(path, data):
    import json
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def multi_sample(tmp_path_factory):
    """Events with several primaries from a --primaries file (C++ core)."""
    folder = tmp_path_factory.mktemp("multi")
    primaries = write_json(folder / "events.json", MULTI_EVENTS)
    return simulate(folder / "multi.h5", "--primaries", primaries, "--seed", 11, *GEOMETRY_ARGS)


@pytest.fixture(scope="session")
def pybind_multi_sample(tmp_path_factory):
    """The same events through geant4_pybind (needs geant4_pybind)."""
    pytest.importorskip("geant4_pybind")
    folder = tmp_path_factory.mktemp("pybind_multi")
    primaries = write_json(folder / "events.json", MULTI_EVENTS)
    return simulate(folder / "multi.h5", "--primaries", primaries, "--seed", 11, *GEOMETRY_ARGS,
                    "--backend", "geant4_pybind")


@pytest.fixture(scope="session")
def listed_shower_sample(tmp_path_factory):
    """The showers of shower_sample, given as a one-particle --primaries file."""
    folder = tmp_path_factory.mktemp("listed")
    one = [{"pdgid": 11, "energy_GeV": 1.0, "position_m": list(SHOWER_START), "direction": [0, 0.6, 0.8]}]
    primaries = write_json(folder / "electron.json", one)
    return simulate(folder / "electron.h5", "--primaries", primaries, "-n", 3, "--seed", 3,
                    *GEOMETRY_ARGS)
