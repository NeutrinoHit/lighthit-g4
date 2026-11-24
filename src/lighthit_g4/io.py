"""HDF5 layout read by LightHit.

``/header`` (attribute ``json``): package and Geant4 versions, configuration,
thresholds and production cuts.  For every event ``n``:

* ``event_<n>/tracks``: charged-particle vertices, numeric columns only
  (``TRACK_DTYPE``), one contiguous block per ``uid``;
* ``event_<n>/particles``: every new track with parent, creator process,
  start, direction, kinetic energy and whether it was tracked;
* attributes of ``event_<n>``: primary, seed, CPU time.
"""
import json
from pathlib import Path

import h5py
import numpy as np

from .simulation import Event, Primary

__all__ = ["Writer", "read_event", "read_header", "event_count"]


class Writer:
    """Events of one :class:`Simulation`; the header is written on close,
    when the production cuts of the run are known."""

    def __init__(self, path, simulation):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.simulation = simulation
        self.file = h5py.File(self.path, "w")
        self.count = 0

    def write(self, event: Event):
        group = self.file.create_group(f"event_{self.count}")
        group.create_dataset("tracks", data=event.tracks)
        group.create_dataset("particles", data=event.particles)
        primary = event.primary
        group.attrs["primary"] = json.dumps({
            "pdgid": int(primary.pdgid), "energy_GeV": float(primary.energy_GeV),
            "position_m": list(primary.position_m), "direction": list(primary.direction),
            "time_ns": float(primary.time_ns), "polarization": float(primary.polarization)})
        group.attrs["seed"] = int(event.seed)
        group.attrs["cpu_s"] = float(event.cpu_s)
        group.attrs["info"] = json.dumps(event.info)
        self.count += 1

    def close(self):
        self.file.attrs["json"] = json.dumps(self.simulation.header())
        self.file.attrs["events"] = self.count
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_header(path):
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["json"])


def event_count(path):
    with h5py.File(path, "r") as f:
        return int(f.attrs.get("events", sum(k.startswith("event_") for k in f)))


def read_event(path, event=0):
    with h5py.File(path, "r") as f:
        group = f[f"event_{int(event)}"]
        primary = json.loads(group.attrs["primary"])
        return Event(int(event), Primary(**primary), int(group.attrs["seed"]),
                     np.asarray(group["tracks"][()]), np.asarray(group["particles"][()]),
                     float(group.attrs["cpu_s"]), json.loads(group.attrs.get("info", "{}")))
