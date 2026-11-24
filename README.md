# lighthit-g4

A small Geant4 application driven from Python: one particle is shot into a
cylinder of water, every charged step is recorded, and the event is written
in the track format that [LightHit](https://github.com/NeutrinoHit/LightHit)
reads. It is meant for tau-neutrino studies in Baikal-GVD without a full
detector-simulation framework.

The event loop and the recording run in C++ (`lighthit_g4._core`, Geant4
11.4.1 linked in); configuration, files and analysis stay in Python. The same
application with the user actions in Python, through
[geant4_pybind](https://pypi.org/project/geant4-pybind/), is kept as a
reference: on the same seeds both give identical events (tested).

## Install

```bash
python -m pip install lighthit-g4               # wheel with Geant4 inside
python -m pip install 'lighthit-g4[pythia]'     # + Pythia 8 tau decays
```

Wheels are built for Linux (x86_64, aarch64) and macOS (x86_64, arm64),
Python 3.12 and newer. Geant4 data sets (about 2 GB) are taken from the
`G4...DATA` environment variables, `$LIGHTHIT_G4_DATA`, `~/.geant4_pybind`
or `~/.lighthit-g4/geant4-data`; missing ones are downloaded there from the
Geant4 data server on first use and checked against their MD5.

From source, Geant4 must first be built with static libraries:

```bash
ci/build_geant4.sh $HOME/geant4-static           # Geant4 11.4.1, about 130 MB
CMAKE_PREFIX_PATH=$HOME/geant4-static python -m pip install '.[pythia,test]'
```

## Run

```bash
lighthit-g4 --pdg 11 --energy-GeV 10 --position 0 0 -100 --direction 0 0 1 -o e-10GeV.h5
lighthit-g4 --pdg 15 --energy-GeV 1000 --tau-decayer pythia --polarization -1 -n 10 -o tau.h5
```

```python
from lighthit_g4 import Config, Primary, Simulation, Writer

sim = Simulation(Config(tau_decayer="pythia"))      # one per process
with Writer("tau.h5", sim) as out:
    for _ in range(10):
        out.write(sim.run(Primary(pdgid=15, energy_GeV=1000.0, polarization=-1.0)))
```

Read back with `read_event(path, n)`, or give the file to LightHit:
`lighthit.sources.G4Shower.from_hdf5(path, n)`.

Options: physics list (any Geant4 reference list, default `FTFP_BERT`),
water radius and height, position of the water centre in the user frame,
range cut (default 0.5 mm), Cherenkov index for the kill rule (default 1.35),
`--secondaries none` (only the primary is tracked; its secondaries are
listed), `--keep-below-threshold`, `--tau-decayer geant4|pythia`,
`--polarization` (helicity of the primary, used by external decays),
`--backend core|geant4_pybind`.

## File layout

`/` attribute `json`: package and Geant4 versions, configuration, light
threshold, production-cut energies, the external decayer and its counts,
and, for every primary species of the run, its Geant4 mass, charge,
lifetime, decay table and decayer.

`event_<n>/tracks` — charged vertices, numeric columns only:
`uid, parent_uid, pdgid, x_m, y_m, z_m, t_ns, beta, step_length_m, e_kin_GeV`.
Rows of one `uid` are contiguous and in time order; the first row is the
particle start (`step_length_m = 0`), each next row ends a step and carries
its true path length. At-rest steps (zero length) are not written.

`event_<n>/particles` — every new track: `uid, parent_uid, pdgid, mass_GeV,
creator, x_m, y_m, z_m, t_ns, dx, dy, dz, e_kin_GeV, tracked`. Neutrinos are
listed but not tracked. Attributes of `event_<n>`: primary, seed, CPU time,
`info` (number of electrons and gammas killed at birth below the light
threshold, which are not listed).

Units: m, ns, GeV. Positions are in the user frame (`center_m` is the centre
of the water cylinder).

## Light threshold

With index `n`, a particle of mass `M` radiates only above the kinetic energy
`T = M (1/sqrt(1 - 1/n^2) - 1)`; for an electron and `n = 1.35` it is
0.2497 MeV (electron mass from Geant4). The run kills

* electrons below `T_e` (at birth or while slowing down): every photon or
  delta electron they can still produce carries less than their kinetic energy;
* gammas below `T_e`: a Compton or photoelectron gets less than the gamma energy.

Nothing else is killed: positrons annihilate into gammas that can make
visible electrons, and hadrons and nuclei below their own threshold can
still produce visible gammas and pions in nuclear interactions. With the
0.5 mm cut, Geant4 produces delta electrons from 0.2239 MeV, below `T_e`.

Check (300 electrons of 3 GeV per mode, LightHit `load_event` photon count):
with the kill rule 150024 ± 244 photons per event, without it 150141 ± 253;
the difference is −0.08 % ± 0.23 %, the event-to-event spread is the same
(2.8 % and 2.9 %).

## Tau decays

`tau_decayer="geant4"` (default) uses the Geant4 11.4.1 decay table of
`tau-`, stored in the file header:

| BR | model | daughters |
|---|---|---|
| 0.2541 | Phase Space | π⁰ π⁻ ν_τ |
| 0.1784 | Tau Leptonic Decay | e⁻ ν̄_e ν_τ |
| 0.1736 | Tau Leptonic Decay | μ⁻ ν̄_μ ν_τ |
| 0.1106 | Phase Space | π⁻ ν_τ |
| 0.0946 | Phase Space | π⁻ π⁻ π⁺ ν_τ |
| 0.0917 | Phase Space | π⁰ π⁰ π⁻ ν_τ |

The listed BR add up to 0.903; the sampled frequencies follow BR / 0.903
(tested). The hadronic channels use phase space, without resonance
structure, and the tau is unpolarized. Geant4 uses m_τ = 1776.86 MeV and
τ = 290.3 fs; PDG values through the `particle` package are 1776.93 MeV and
290.34 fs.

`tau_decayer="pythia"` hands tau decays to Pythia 8 (package `pythia8mc`,
installed separately). Geant4 still transports the tau; at the decay point
Pythia gets the tau four-momentum and its helicity (`Primary.polarization`
of a primary tau), decays the tau and the short-lived resonances, and returns
every product that Geant4 tracks (π⁰, η, kaons, ...) undecayed. Pythia takes
an external tau helicity as −1 or +1, so each decay draws +1 with
probability (1 + h)/2. The random numbers come from the Geant4 engine, so an
event is fixed by its seed. Test: in τ → π ν, ⟨E_π/E_τ⟩ = 1/2 + h/6 (V−A,
collinear limit) for h = ±1.

Visible energy of the tau decay (all products except neutrinos and muons) as
a fraction y of the tau energy, 100 GeV taus, Pythia 8.318 with forced
polarization `TauDecays:tauPolarization`:

| | Geant4 | Pythia, h = 0 | Pythia, h = −1 | Pythia, h = +1 |
|---|---|---|---|---|
| fraction τ → e ν ν | 0.200 | 0.178 | 0.176 | 0.177 |
| fraction τ → μ ν ν | 0.189 | 0.171 | 0.175 | 0.173 |
| ⟨y⟩, τ → e ν ν | 0.349 | 0.349 | 0.399 | 0.299 |
| ⟨y⟩, hadronic decays | 0.684 | 0.643 | 0.589 | 0.701 |
| ⟨y⟩, all decays | 0.488 | 0.481 | 0.452 | 0.509 |

(20000 Geant4 and 100000 Pythia decays per column; errors of the means
≤ 0.004.) A tau⁻ from a ν_τ charged-current interaction at E ≫ m_τ is
left-handed (h → −1); its polarization should come from the neutrino event
generator together with the tau.

## Tests

```bash
python -m pytest -q
```

Tau (Geant4 table, 2000 decays at 100 GeV): mean decay length against γβcτ
from PDG, channel frequencies against the Geant4 table, energy and momentum
of the decay products. Tau (Pythia, 5000 decays per helicity): pion spectrum
against the helicity, energy of the products, products left to Geant4.
Format: row blocks, frame, cuts, kill rule, that LightHit reads the file and
finds the shower axis along the primary direction, and that the C++ core and
the geant4_pybind version give identical events.

## Speed

Apple M4 Pro, one core, FTFP_BERT, cut 0.5 mm, kill rule on; both backends
produce the same events:

| primary | C++ core | geant4_pybind | rows per event |
|---|---|---|---|
| e⁻ 1 GeV | 1.0 ms | 9.4 ms | 1.3 × 10³ |
| e⁻ 10 GeV | 10 ms | 96 ms | 1.3 × 10⁴ |
| e⁻ 100 GeV | 0.10 s | 0.96 s | 1.3 × 10⁵ |
| e⁻ 1 TeV | 0.99 s | 9.8 s | 1.3 × 10⁶ |
| τ 1 TeV (full event, 30 events) | 0.43 s | 4.0 s | 5.3 × 10⁵ |

(`benchmarks/speed.py`; the first event of a run is left out.) Without the
kill rule an electron shower takes about 1.9 times longer and writes 3.3
times more rows.

The cost and the file size (about 0.1 MB of tracks per GeV) grow linearly
with the shower energy. Above about 10 TeV the electromagnetic part has to be
replaced by stored showers placed at their starting points; this is not done
yet.

## Licenses

lighthit-g4 is MIT-licensed. The wheels contain Geant4 (Geant4 Software
License) with its bundled CLHEP (LGPL-3.0), Expat, PTL, zlib and g4tools,
and nanobind; their license texts are in `LICENSES/`. Pythia 8 (GPL-2.0) is
not part of the wheels: it is installed separately as `pythia8mc`.
