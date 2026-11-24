"""Tau propagation and decay against PDG (``particle``) and Geant4's own decay table."""
import collections
import math

import numpy as np
import pytest

from lighthit_g4 import event_count, read_event, read_header
from lighthit_g4.decays import decay_products, decay_vertex, tau_decay_mode

from conftest import PYTHIA_EVENTS, TAU_ENERGY_GEV, TAU_EVENTS

# Geant4 11.4 tau- channels by daughters (the decay table stored in the file
# header); the test compares sampled frequencies with them.
NAMES_TO_PDG = {"pi0": 111, "pi-": -211, "pi+": 211, "nu_tau": 16, "e-": 11,
                "anti_nu_e": -12, "mu-": 13, "anti_nu_mu": -14}


@pytest.fixture(scope="module")
def events(tau_sample):
    assert event_count(tau_sample) == TAU_EVENTS
    return [read_event(tau_sample, i) for i in range(TAU_EVENTS)]


def test_every_tau_decays_once(events):
    for event in events:
        assert tau_decay_mode(event) is not None
        tau_rows = event.tracks[event.tracks["uid"] == 1]
        assert np.all(tau_rows["pdgid"] == 15)


def test_mean_decay_length_matches_pdg(events):
    from particle import Particle
    tau = Particle.from_pdgid(15)
    mass_GeV = tau.mass / 1e3
    p_GeV = math.sqrt(TAU_ENERGY_GEV ** 2 + 2 * TAU_ENERGY_GEV * mass_GeV)
    expected_m = p_GeV / mass_GeV * tau.ctau * 1e-3      # gamma beta c tau
    lengths = np.array([np.linalg.norm(decay_vertex(e)[0]) for e in events])
    sigma = expected_m / math.sqrt(len(lengths))          # exponential: sd = mean
    assert abs(lengths.mean() - expected_m) < 4 * sigma, (lengths.mean(), expected_m)


def test_decay_products_conserve_energy_and_momentum(events, tau_sample):
    header = read_header(tau_sample)
    assert header["config"]["secondaries"] == "none"
    tau_mass = header["primaries"]["15"]["mass_MeV"] / 1e3
    for event in events:
        rows = event.tracks[event.tracks["uid"] == 1]
        last = rows[-1]
        daughters = decay_products(event, 1)
        energy = daughters["e_kin_GeV"] + daughters["mass_GeV"]
        tau_energy = last["e_kin_GeV"] + tau_mass
        # measured on 2000 decays: |residual| <= 2e-8 relative (2 keV at 100 GeV),
        # daughters slightly below the stored end-of-step tau energy
        assert abs(energy.sum() - tau_energy) < 1e-7 * tau_energy
        kinetic = daughters["e_kin_GeV"]
        p = np.sqrt(kinetic * (kinetic + 2 * daughters["mass_GeV"]))
        momentum = (p[:, None] * np.column_stack([daughters["dx"], daughters["dy"], daughters["dz"]])).sum(0)
        tau_p = math.sqrt(last["e_kin_GeV"] * (last["e_kin_GeV"] + 2 * tau_mass))
        chord = np.array([last["x_m"] - rows[-2]["x_m"], last["y_m"] - rows[-2]["y_m"],
                          last["z_m"] - rows[-2]["z_m"]])
        assert abs(np.linalg.norm(momentum) - tau_p) < 1e-7 * tau_p
        # the stored chord misses the multiple-scattering kink at the end of the
        # step, so only the angle is compared, against a loose bound
        cosine = momentum @ chord / (np.linalg.norm(momentum) * np.linalg.norm(chord))
        assert cosine > math.cos(1e-3)


def test_branching_matches_geant4_table(events, tau_sample):
    channels = read_header(tau_sample)["primaries"]["15"]["decay_table"]
    expected = {tuple(sorted(NAMES_TO_PDG[n] for n in c["daughters"])): c["br"] for c in channels}
    norm = sum(expected.values())
    seen = collections.Counter()
    for event in events:
        decayed = decay_products(event, 1)
        seen[tuple(sorted(int(p) for p in decayed["pdgid"]))] += 1
    assert set(seen) <= set(expected), set(seen) - set(expected)
    n = len(events)
    for key, br in expected.items():
        p = br / norm
        assert abs(seen[key] - n * p) < 4 * math.sqrt(n * p * (1 - p)), (key, seen[key], n * p)


def test_decay_modes_are_classified(events):
    modes = collections.Counter(tau_decay_mode(e) for e in events)
    assert set(modes) == {"e", "mu", "hadrons"}


def test_pythia_pion_spectrum_follows_helicity(pythia_sample):
    """tau -> pi nu: in the tau rest frame dN/dcos = (1 + h cos)/2 (V-A, spin-0 pion,
    left-handed neutrino); in the collinear limit x = E_pi/E_tau = (1 + cos)/2
    up to m_pi^2/m_tau^2, so <x> = 1/2 + h/6."""
    path, helicity = pythia_sample
    header = read_header(path)
    assert header["primaries"]["15"]["decayer"] == "external"
    assert header["primaries"]["15"]["decay_table"] == []
    assert header["external_decays"]["decays"] == PYTHIA_EVENTS
    assert header["external_decays"]["unknown_products"] == 0
    assert header["external_decays"]["decayer"].startswith("Pythia 8.")
    x = []
    for i in range(event_count(path)):
        decayed = decay_products(read_event(path, i), 1)
        visible = decayed[~np.isin(np.abs(decayed["pdgid"]), (12, 14, 16))]
        if len(visible) == 1 and abs(visible["pdgid"][0]) == 211:
            energy = decayed["e_kin_GeV"] + decayed["mass_GeV"]
            x.append((visible["e_kin_GeV"][0] + visible["mass_GeV"][0]) / energy.sum())
    x = np.array(x)
    expected = 0.5 + helicity / 6
    assert abs(x.mean() - expected) < 4 * x.std() / math.sqrt(len(x)), (x.mean(), expected)


def test_pythia_products_conserve_energy_and_leave_pi0_to_geant4(pythia_sample):
    path, _ = pythia_sample
    tau_mass = read_header(path)["primaries"]["15"]["mass_MeV"] / 1e3
    seen = collections.Counter()
    for i in range(event_count(path)):
        event = read_event(path, i)
        decayed = decay_products(event, 1)
        seen.update(int(abs(p)) for p in decayed["pdgid"])
        last = event.tracks[event.tracks["uid"] == 1][-1]
        energy = (decayed["e_kin_GeV"] + decayed["mass_GeV"]).sum()
        # daughters keep the Pythia momenta with Geant4 masses
        assert abs(energy - (last["e_kin_GeV"] + tau_mass)) < 1e-5 * energy
    assert seen[111] > 0
    assert set(seen) <= {11, 12, 13, 14, 16, 22, 111, 130, 211, 221, 310, 311, 321}, set(seen)
