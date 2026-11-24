"""``lighthit-g4``: simulate events and write an HDF5 file.

The primaries of an event are one particle (``--pdg``, ``--energy-GeV``, ...)
or a JSON file (``--primaries``).  The file holds either one event, a list of
particles, which is simulated ``-n`` times, or a list of events, each a list
of particles, simulated once each.  A particle is an object with the fields of
``Primary``: ``pdgid``, ``energy_GeV`` (kinetic) and optionally
``position_m``, ``direction``, ``time_ns``, ``polarization``.
"""
import argparse
import json
import sys
import time

from .simulation import Config, Primary, Simulation


def parser():
    p = argparse.ArgumentParser(prog="lighthit-g4", description=__doc__)
    p.add_argument("-n", "--events", type=int, default=1)
    p.add_argument("-o", "--output", required=True, help="HDF5 file to write")
    p.add_argument("--pdg", type=int, help="PDG code of the primary")
    p.add_argument("--energy-GeV", type=float, help="kinetic energy")
    p.add_argument("--primaries", metavar="JSON",
                   help="file with the primaries of one event or of a list of events")
    p.add_argument("--position", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="start in m, user frame")
    p.add_argument("--direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("DX", "DY", "DZ"))
    p.add_argument("--polarization", type=float, default=0.0,
                   help="longitudinal polarization (helicity) of the primary, for external decays")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--secondaries", choices=("all", "none"), default=Config.secondaries)
    p.add_argument("--physics", default=Config.physics_list)
    p.add_argument("--tau-decayer", choices=("geant4", "pythia"), default=Config.tau_decayer)
    p.add_argument("--backend", choices=("core", "geant4_pybind"), default=Config.backend)
    p.add_argument("--cut-m", type=float, default=Config.production_cut_m, help="range cut")
    p.add_argument("--cherenkov-index", type=float, default=Config.cherenkov_index)
    p.add_argument("--keep-below-threshold", action="store_true",
                   help="track every particle, also those that cannot radiate")
    p.add_argument("--radius-m", type=float, default=Config.water_radius_m)
    p.add_argument("--height-m", type=float, default=Config.water_height_m)
    p.add_argument("--center", type=float, nargs=3, default=Config.center_m, metavar=("X", "Y", "Z"),
                   help="water centre in the user frame, m")
    p.add_argument("--verbose", type=int, default=0)
    return p


def read_primaries(path):
    """Events from a ``--primaries`` file: (list of events, True if it held one event)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list")
    if all(isinstance(item, dict) for item in data):
        return [tuple(Primary(**item) for item in data)], True
    if all(isinstance(item, list) for item in data):
        return [tuple(Primary(**item) for item in event) for event in data], False
    raise ValueError(f"{path}: expected a list of particles or a list of events")


def main(argv=None):
    from .io import Writer
    p = parser()
    args = p.parse_args(argv)
    if args.primaries is not None:
        if args.pdg is not None or args.energy_GeV is not None:
            p.error("--primaries replaces --pdg and --energy-GeV")
        events, single = read_primaries(args.primaries)
        if single:
            events = events * args.events
        elif args.events != 1:
            p.error("-n repeats a one-event --primaries file; this file holds a list of events")
    elif args.pdg is None or args.energy_GeV is None:
        p.error("give --pdg and --energy-GeV, or --primaries")
    else:
        events = [Primary(args.pdg, args.energy_GeV, tuple(args.position), tuple(args.direction),
                          polarization=args.polarization)] * args.events
    config = Config(physics_list=args.physics, water_radius_m=args.radius_m,
                    water_height_m=args.height_m, center_m=tuple(args.center),
                    production_cut_m=args.cut_m, cherenkov_index=args.cherenkov_index,
                    kill_below_threshold=not args.keep_below_threshold,
                    secondaries=args.secondaries, tau_decayer=args.tau_decayer,
                    backend=args.backend, seed=args.seed)
    sim = Simulation(config, verbose=args.verbose)
    start = time.perf_counter()
    with Writer(args.output, sim) as writer:
        for primaries in events:
            writer.write(sim.run(primaries))
    summary = {"output": args.output, "events": len(events),
               "wall_s": round(time.perf_counter() - start, 3)}
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
