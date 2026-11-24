"""One Geant4 application: a water cylinder, a particle gun, recorded steps.

An event starts from one primary or from a list of primaries (for example the
final state of a neutrino interaction from an event generator): each primary
is its own vertex with its own start, time and polarization, and Geant4
numbers them ``uid = 1, 2, ...`` in the order of the list (``parent_uid = 0``).

Two backends run the same application with the same rules: the C++ core
(``backend="core"``, Geant4 linked into ``lighthit_g4._core``) and the
geant4_pybind version with the user actions in Python
(``backend="geant4_pybind"``), kept as a reference.

The world is a cylinder of NIST water (``G4_WATER``) with a vertical axis.
Every charged step is recorded as two vertices of its particle: the start of
the first step (``step_length_m = 0``) and the end of every step.  This is the
row layout that LightHit reads (``lighthit.sources.G4Shower.from_hdf5``):
consecutive rows of one ``uid`` form a step, and the stored true path length
belongs to the row that ends the step.

Every new track is listed in the particle table with its parent, creator
process and start, also when it is killed at once (neutrinos, or all
secondaries with ``secondaries="none"``), so that the history of any
particle (production, decay vertex, decay products, neutrinos) can be read back.  The
only tracks left out are the electrons and gammas killed below the light
threshold (see below): they are counted in ``Event.info``.

Light thresholds.  A particle radiates Cherenkov light only while
``beta * n > 1``.  With ``cherenkov_index = n`` the kinetic threshold of a
particle of mass ``M`` is ``T = M (1 / sqrt(1 - 1/n^2) - 1)``.  The run kills

* an electron (new or slowing down) below the electron threshold ``T_e``:
  its own light is zero and every photon or delta electron it can still make
  carries less than its kinetic energy, so it is below ``T_e`` too;
* a gamma with energy below ``T_e``: the Compton or photoelectron it can make
  has a kinetic energy below the gamma energy.

Nothing else is killed.  Positrons are kept to the end: the annihilation
gammas can still make visible Compton electrons.  Protons, nuclei and other
hadrons below their own threshold are kept as well: they do not radiate, but
their nuclear interactions can still make visible gammas and pions.
``cherenkov_index`` must not be smaller than the largest
phase index used downstream; the default equals the LightHit default
``reference_phase_index``.

Tau decays.  ``tau_decayer="geant4"`` uses the Geant4 decay table (phase
space for the hadronic channels, no polarization); ``tau_decayer="pythia"``
hands tau decays to Pythia 8 (``lighthit_g4.pythia``) with the longitudinal
polarization of the tau (``Primary.polarization`` for a primary tau).
"""
from dataclasses import asdict, dataclass, field
import math

import numpy as np

from ._version import VERSION

__all__ = ["Config", "Primary", "Event", "Simulation", "TRACK_DTYPE",
           "PARTICLE_DTYPE", "kinetic_threshold_MeV"]

NEUTRINOS = {12, 14, 16}

TRACK_DTYPE = np.dtype([
    ("uid", "<i8"), ("parent_uid", "<i8"), ("pdgid", "<i8"),
    ("x_m", "<f8"), ("y_m", "<f8"), ("z_m", "<f8"), ("t_ns", "<f8"),
    ("beta", "<f8"), ("step_length_m", "<f8"), ("e_kin_GeV", "<f8"),
])

PARTICLE_DTYPE = np.dtype([
    ("uid", "<i8"), ("parent_uid", "<i8"), ("pdgid", "<i8"), ("mass_GeV", "<f8"),
    ("creator", "S24"),
    ("x_m", "<f8"), ("y_m", "<f8"), ("z_m", "<f8"), ("t_ns", "<f8"),
    ("dx", "<f8"), ("dy", "<f8"), ("dz", "<f8"),
    ("e_kin_GeV", "<f8"), ("tracked", "?"),
])


def kinetic_threshold_MeV(mass_MeV, index):
    """Kinetic energy at which ``beta * index = 1``."""
    if index <= 1:
        raise ValueError("the Cherenkov index must exceed 1")
    return mass_MeV * (1.0 / math.sqrt(1.0 - 1.0 / index ** 2) - 1.0)


@dataclass(frozen=True)
class Config:
    """Fixed for the whole process: Geant4 allows one run manager."""
    physics_list: str = "FTFP_BERT"
    water_radius_m: float = 1000.0
    water_height_m: float = 1500.0
    center_m: tuple = (0.0, 0.0, 0.0)   # user coordinates of the cylinder centre
    production_cut_m: float = 0.5e-3    # range cut for gamma, e-, e+, proton
    cherenkov_index: float = 1.35
    kill_below_threshold: bool = True
    secondaries: str = "all"            # "all" or "none" (listed, not tracked)
    tau_decayer: str = "geant4"         # "geant4" (decay table) or "pythia"
    backend: str = "core"               # "core" (C++) or "geant4_pybind"
    seed: int = 1

    def __post_init__(self):
        if self.secondaries not in ("all", "none"):
            raise ValueError("secondaries must be 'all' or 'none'")
        if self.tau_decayer not in ("geant4", "pythia"):
            raise ValueError("tau_decayer must be 'geant4' or 'pythia'")
        if self.backend not in ("core", "geant4_pybind"):
            raise ValueError("backend must be 'core' or 'geant4_pybind'")
        if self.backend == "geant4_pybind" and self.tau_decayer != "geant4":
            raise ValueError("the geant4_pybind backend has only the Geant4 decay tables")
        if len(tuple(self.center_m)) != 3:
            raise ValueError("center_m must have three components")
        for name in ("water_radius_m", "water_height_m", "production_cut_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        kinetic_threshold_MeV(1.0, self.cherenkov_index)


@dataclass(frozen=True)
class Primary:
    """One particle: PDG code, kinetic energy, start and direction (user frame).

    ``polarization`` is the longitudinal polarization (helicity), from -1 to 1;
    only external decays use it (``tau_decayer="pythia"``).
    """
    pdgid: int
    energy_GeV: float
    position_m: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, 1.0)
    time_ns: float = 0.0
    polarization: float = 0.0

    def __post_init__(self):
        direction = np.asarray(self.direction, float)
        norm = np.linalg.norm(direction)
        if direction.shape != (3,) or not norm > 0:
            raise ValueError("direction must be a nonzero 3-vector")
        object.__setattr__(self, "direction", tuple(direction / norm))
        object.__setattr__(self, "position_m", tuple(float(v) for v in self.position_m))
        if not self.energy_GeV > 0:
            raise ValueError("energy_GeV must be positive")
        if not -1.0 <= self.polarization <= 1.0:
            raise ValueError("polarization must lie in [-1, 1]")


def as_primaries(primaries):
    """A :class:`Primary` or an iterable of them, as a non-empty tuple."""
    if isinstance(primaries, Primary):
        return (primaries,)
    out = tuple(primaries)
    if not out:
        raise ValueError("an event needs at least one primary")
    for primary in out:
        if not isinstance(primary, Primary):
            raise TypeError(f"expected Primary, got {type(primary).__name__}")
    return out


@dataclass
class Event:
    """One event: ``primaries[k]`` has ``uid = k + 1``."""
    index: int
    primaries: tuple
    seed: int
    tracks: np.ndarray
    particles: np.ndarray
    cpu_s: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def primary(self):
        """The primary of a one-particle event."""
        if len(self.primaries) != 1:
            raise ValueError(f"the event has {len(self.primaries)} primaries; use Event.primaries")
        return self.primaries[0]

    def particle(self, uid):
        row = self.particles[self.particles["uid"] == uid]
        if len(row) != 1:
            raise KeyError(uid)
        return row[0]

    def daughters(self, uid):
        return self.particles[self.particles["parent_uid"] == uid]


class _CoreEngine:
    """The C++ engine with the interface of the geant4_pybind one."""

    def __init__(self, config, verbose=0):
        from . import _core
        from .datasets import ensure_datasets
        ensure_datasets(verbose=verbose)
        settings = _core.Settings()
        settings.physics_list = config.physics_list
        settings.water_radius_m = config.water_radius_m
        settings.water_height_m = config.water_height_m
        settings.center_m = tuple(float(v) for v in config.center_m)
        settings.production_cut_m = config.production_cut_m
        settings.cherenkov_index = config.cherenkov_index
        settings.kill_below_threshold = config.kill_below_threshold
        settings.track_secondaries = config.secondaries == "all"
        settings.verbose = int(verbose)
        self.decayer = None
        external = []
        if config.tau_decayer == "pythia":
            from .pythia import PythiaDecays
            self.decayer = PythiaDecays(verbose=verbose)
            external = [15, -15]
        self.engine = _core.Engine(settings, self.decayer, external)
        if self.decayer is not None:
            self.decayer.leave_to_geant4(self.engine.tracked_species())
        self.geant4_version = _core.geant4_version
        self.electron_mass_MeV = self.engine.electron_mass_MeV
        self.threshold_e_MeV = self.engine.threshold_e_MeV
        self._core = _core

    def run(self, primaries, seed):
        shots = []
        for primary in primaries:
            shot = self._core.Shot()
            shot.pdgid = int(primary.pdgid)
            shot.energy_GeV = float(primary.energy_GeV)
            shot.position_m = primary.position_m
            shot.direction = primary.direction
            shot.time_ns = float(primary.time_ns)
            shot.polarization = float(primary.polarization)
            shots.append(shot)
        tracks, particles, below, cpu = self.engine.run(shots, int(seed))
        return (np.frombuffer(tracks, TRACK_DTYPE).copy(),
                np.frombuffer(particles, PARTICLE_DTYPE).copy(), below, cpu)

    def energy_cuts_MeV(self):
        return self.engine.energy_cuts_MeV()

    def particle_properties(self, pdgid):
        return self.engine.particle_properties(int(pdgid))

    def decayer_counts(self):
        counts = self.engine.decayer_counts()
        if self.decayer is not None:
            counts["decayer"] = self.decayer.description
        return counts


class Simulation:
    """Geant4 set up once; :meth:`run` simulates one event from one or more primaries."""
    _instance = None

    def __init__(self, config=Config(), *, verbose=0):
        if Simulation._instance is not None:
            raise RuntimeError("Geant4 allows one Simulation per process")
        Simulation._instance = self
        self.config = config
        if config.backend == "core":
            self.engine = _CoreEngine(config, verbose)
        else:
            from ._g4py import PybindEngine
            self.engine = PybindEngine(config, verbose)
        self.electron_mass_MeV = self.engine.electron_mass_MeV
        self.threshold_e_MeV = self.engine.threshold_e_MeV
        self.events_done = 0
        self.primary_species = set()
        self.energy_cuts_MeV = None      # the cuts table is built by the first event

    def particle_properties(self, pdgid):
        """Geant4 definition of one particle: mass, charge, lifetime, decay table
        and who decays it (``"Geant4"`` or ``"external"``)."""
        return self.engine.particle_properties(pdgid)

    def header(self):
        return {
            "package": "lighthit-g4", "version": VERSION,
            "geant4": self.engine.geant4_version.strip("$").replace("Name:", "").strip(),
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in asdict(self.config).items()},
            "electron_mass_MeV": self.electron_mass_MeV,
            "cherenkov_threshold_e_kinetic_MeV": self.threshold_e_MeV,
            "production_cut_energy_MeV": self.energy_cuts_MeV,
            # every primary species shot in this run, as Geant4 defines it
            "primaries": {str(p): self.particle_properties(p) for p in sorted(self.primary_species)},
            "external_decays": self.engine.decayer_counts(),
            "track_columns": list(TRACK_DTYPE.names),
            "particle_columns": list(PARTICLE_DTYPE.names),
            "units": "m, ns, GeV; positions in the user frame (center_m is the water centre)",
        }

    def run(self, primaries, *, seed=None):
        """Simulate one event from a :class:`Primary` or a list of them (all
        shot into the same event, ``uid`` 1, 2, ... in list order); the seed
        defaults to ``config.seed + event index``."""
        primaries = as_primaries(primaries)
        index = self.events_done
        seed = self.config.seed + index if seed is None else int(seed)
        tracks, particles, below, cpu = self.engine.run(primaries, seed)
        if self.energy_cuts_MeV is None:
            self.energy_cuts_MeV = self.engine.energy_cuts_MeV()
        self.events_done += 1
        self.primary_species.update(int(p.pdgid) for p in primaries)
        info = {"killed_below_threshold_at_birth": int(below)}
        return Event(index, primaries, seed, tracks, particles, cpu, info)
