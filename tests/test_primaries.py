"""Events with several primaries: one vertex per particle, uid 1..N in list order."""
import numpy as np
import pytest

from lighthit_g4 import event_count, read_event, read_header

from conftest import MULTI_EVENTS


def same_events(path_a, path_b):
    assert event_count(path_a) == event_count(path_b)
    for i in range(event_count(path_a)):
        a, b = read_event(path_a, i), read_event(path_b, i)
        assert a.seed == b.seed
        for name in ("tracks", "particles"):
            x, y = getattr(a, name), getattr(b, name)
            assert x.shape == y.shape
            for column in x.dtype.names:
                assert np.array_equal(x[column], y[column]), (i, name, column)


def test_primaries_are_tracks_one_to_n(multi_sample):
    assert event_count(multi_sample) == len(MULTI_EVENTS)
    for index, given in enumerate(MULTI_EVENTS):
        event = read_event(multi_sample, index)
        assert [p.pdgid for p in event.primaries] == [p["pdgid"] for p in given]
        first = event.particles[event.particles["parent_uid"] == 0]
        assert list(first["uid"]) == list(range(1, len(given) + 1))
        for row, primary in zip(first, event.primaries):
            assert row["pdgid"] == primary.pdgid
            assert row["creator"] == b"primary"
            assert row["e_kin_GeV"] == pytest.approx(primary.energy_GeV)
            assert (row["x_m"], row["y_m"], row["z_m"]) == pytest.approx(primary.position_m)
            assert (row["dx"], row["dy"], row["dz"]) == pytest.approx(primary.direction)
            assert row["t_ns"] == pytest.approx(primary.time_ns)
        # every other particle descends from one of the primaries
        parents = dict(zip(event.particles["uid"], event.particles["parent_uid"]))
        for uid in event.particles["uid"]:
            while parents[uid] != 0:
                uid = parents[uid]
            assert 1 <= uid <= len(given)


def test_polarization_and_species_are_kept(multi_sample):
    event = read_event(multi_sample, 0)
    assert event.primaries[0].polarization == -1.0
    with pytest.raises(ValueError):
        event.primary
    assert read_event(multi_sample, 1).primary.pdgid == 22
    species = {int(k) for k in read_header(multi_sample)["primaries"]}
    assert species == {p["pdgid"] for event in MULTI_EVENTS for p in event}


def test_one_particle_list_equals_single_primary(shower_sample, listed_shower_sample):
    same_events(shower_sample, listed_shower_sample)


def test_core_and_geant4_pybind_agree_on_several_primaries(multi_sample, pybind_multi_sample):
    same_events(multi_sample, pybind_multi_sample)
