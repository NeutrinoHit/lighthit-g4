"""``lighthit-g4``: shoot one kind of primary N times and write an HDF5 file."""
import argparse
import json
import sys
import time

from .simulation import Config, Primary, Simulation


def parser():
    p = argparse.ArgumentParser(prog="lighthit-g4", description=__doc__)
    p.add_argument("-n", "--events", type=int, default=1)
    p.add_argument("-o", "--output", required=True, help="HDF5 file to write")
    p.add_argument("--pdg", type=int, required=True, help="PDG code of the primary")
    p.add_argument("--energy-GeV", type=float, required=True, help="kinetic energy")
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


def main(argv=None):
    from .io import Writer
    args = parser().parse_args(argv)
    config = Config(physics_list=args.physics, water_radius_m=args.radius_m,
                    water_height_m=args.height_m, center_m=tuple(args.center),
                    production_cut_m=args.cut_m, cherenkov_index=args.cherenkov_index,
                    kill_below_threshold=not args.keep_below_threshold,
                    secondaries=args.secondaries, tau_decayer=args.tau_decayer,
                    backend=args.backend, seed=args.seed)
    primary = Primary(args.pdg, args.energy_GeV, tuple(args.position), tuple(args.direction),
                      polarization=args.polarization)
    sim = Simulation(config, verbose=args.verbose)
    start = time.perf_counter()
    with Writer(args.output, sim) as writer:
        for _ in range(args.events):
            writer.write(sim.run(primary))
    summary = {"output": args.output, "events": args.events,
               "wall_s": round(time.perf_counter() - start, 3)}
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
