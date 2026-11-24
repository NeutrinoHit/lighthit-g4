"""CPU time, rows and LightHit photon count per event for one primary.

    python benchmarks/speed.py --pdg 11 --energy-GeV 10 -n 20 [--backend geant4_pybind]
        [--keep-below-threshold] [--tau-decayer pythia --polarization -1]

Prints one JSON line. The first event (Geant4 table building) is left out of
the CPU average when there is more than one event.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from lighthit_g4 import Config, Primary, Simulation, Writer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdg", type=int, required=True)
    p.add_argument("--energy-GeV", type=float, required=True)
    p.add_argument("-n", "--events", type=int, default=5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--keep-below-threshold", action="store_true")
    p.add_argument("--backend", choices=("core", "geant4_pybind"), default="core")
    p.add_argument("--tau-decayer", choices=("geant4", "pythia"), default="geant4")
    p.add_argument("--polarization", type=float, default=0.0)
    p.add_argument("-o", "--output", help="HDF5 file (default: a temporary file)")
    args = p.parse_args()

    sim = Simulation(Config(kill_below_threshold=not args.keep_below_threshold, seed=args.seed,
                            backend=args.backend, tau_decayer=args.tau_decayer))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(args.output or Path(tmp) / "events.h5")
        cpu, rows = [], []
        with Writer(path, sim) as out:
            for _ in range(args.events):
                event = sim.run(Primary(args.pdg, args.energy_GeV, polarization=args.polarization))
                out.write(event)
                cpu.append(event.cpu_s)
                rows.append(len(event.tracks))
        photons = None
        try:
            from lighthit.g4_source import load_event
            photons = np.array([load_event(path, i).photons.sum() for i in range(args.events)])
        except ImportError:
            pass
        size_MB = path.stat().st_size / 1e6
    timed = cpu[1:] if len(cpu) > 1 else cpu
    result = {"backend": args.backend, "pdg": args.pdg, "energy_GeV": args.energy_GeV,
              "events": args.events, "tau_decayer": args.tau_decayer,
              "kill_below_threshold": not args.keep_below_threshold,
              "cpu_s_per_event": float(np.mean(timed)), "rows_per_event": float(np.mean(rows)),
              "file_MB_per_event": size_MB / args.events}
    if photons is not None:
        result.update(photons_mean=float(photons.mean()),
                      photons_error=float(photons.std() / np.sqrt(len(photons))),
                      photons_rel_sd=float(photons.std() / photons.mean()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
