"""The file is what LightHit reads; rows, frames and thresholds are as documented."""
import numpy as np
import pytest

from lighthit_g4 import event_count, kinetic_threshold_MeV, read_event, read_header

from conftest import SHOWER_CENTER, SHOWER_START


@pytest.fixture(scope="module")
def events(shower_sample):
    return [read_event(shower_sample, i) for i in range(event_count(shower_sample))]


def test_header(shower_sample):
    header = read_header(shower_sample)
    assert header["geant4"].startswith("geant4-11")
    cuts = header["production_cut_energy_MeV"]
    # delta electrons that can radiate are produced, not absorbed into dE/dx
    assert cuts["e-"] < header["cherenkov_threshold_e_kinetic_MeV"]
    assert header["cherenkov_threshold_e_kinetic_MeV"] == pytest.approx(
        kinetic_threshold_MeV(header["electron_mass_MeV"], header["config"]["cherenkov_index"]))


def test_rows_are_blocks_starting_with_zero_step(events):
    for event in events:
        uid = event.tracks["uid"]
        assert np.all(np.diff(uid) >= 0)
        first = np.r_[True, uid[1:] != uid[:-1]]
        assert np.all(event.tracks["step_length_m"][first] == 0)
        assert np.all(event.tracks["step_length_m"][~first] > 0)
        assert np.all(np.diff(event.tracks["t_ns"])[~first[1:]] >= 0)


def test_tracks_and_particles_agree(events):
    for event in events:
        particles = event.particles
        assert len(np.unique(particles["uid"])) == len(particles)
        tracked = set(particles["uid"][particles["tracked"]])
        assert set(np.unique(event.tracks["uid"])) <= tracked
        lookup = dict(zip(particles["uid"], particles["parent_uid"]))
        for uid, parent in zip(event.tracks["uid"], event.tracks["parent_uid"]):
            assert lookup[uid] == parent


def test_user_frame(events):
    for event in events:
        start = event.tracks[event.tracks["uid"] == 1][0]
        assert (start["x_m"], start["y_m"], start["z_m"]) == pytest.approx(SHOWER_START)
        radius = np.hypot(event.tracks["x_m"] - SHOWER_CENTER[0], event.tracks["y_m"] - SHOWER_CENTER[1])
        assert radius.max() < 50 + 1e-9
        assert np.all(np.abs(event.tracks["z_m"] - SHOWER_CENTER[2]) < 50 + 1e-9)


def test_no_tracked_electron_below_threshold(events, shower_sample):
    threshold = read_header(shower_sample)["cherenkov_threshold_e_kinetic_MeV"] / 1e3
    for event in events:
        electrons = event.particles[(event.particles["pdgid"] == 11) & event.particles["tracked"]]
        assert np.all(electrons["e_kin_GeV"][electrons["uid"] != 1] >= threshold)


def test_lighthit_reads_the_file(shower_sample, events):
    lighthit = pytest.importorskip("lighthit")
    from lighthit.sources import G4Shower
    from lighthit.g4_source import load_event
    for index, event in enumerate(events):
        shower = G4Shower.from_hdf5(shower_sample, index)
        elements = shower.elements
        assert len(elements.uid) > 0
        assert set(np.unique(elements.uid)) <= set(np.unique(event.tracks["uid"]))
        light = load_event(shower_sample, index)
        assert light.photons.sum() > 0
        # the shower runs along the primary direction
        axis = shower.principal_axis
        assert abs(axis @ np.array([0, 0.6, 0.8])) > 0.9


def test_core_and_geant4_pybind_give_identical_events(shower_sample, pybind_shower_sample):
    """Same Geant4 version, physics, rules and seeds: every row must be equal."""
    assert event_count(shower_sample) == event_count(pybind_shower_sample)
    for i in range(event_count(shower_sample)):
        a, b = read_event(shower_sample, i), read_event(pybind_shower_sample, i)
        for name in ("tracks", "particles"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.shape == y.shape
            for column in x.dtype.names:
                assert np.array_equal(x[column], y[column]), (i, name, column)
    ha, hb = read_header(shower_sample), read_header(pybind_shower_sample)
    assert ha["config"]["backend"] == "core" and hb["config"]["backend"] == "geant4_pybind"
    for key in ha:
        if key != "config":
            assert ha[key] == hb[key], key
