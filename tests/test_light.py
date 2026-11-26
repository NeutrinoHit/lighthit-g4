"""The light sink against LightHit's reference on the same steps (lighthit.streamed)."""
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

lh = pytest.importorskip("lighthit")
pytest.importorskip("numba")
from lighthit.streamed import StreamPlan, stream_elements  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"

KERNEL = '''
import numpy as np
import lighthit as lh

def kernel():
    rng = np.random.default_rng(8)
    strings = rng.uniform(-50, 50, size=(5, 2))
    positions = np.array([[x, y, z] for x, y in strings for z in np.arange(-90., 91., 15.)])
    medium = lh.SpectralMedium([350., 450., 610.], [0.06, 0.03, 0.09], [0.03, 0.02, 0.01],
                               [1.352, 1.341, 1.333], [1.40, 1.37, 1.35], g=0.9)
    detector = lh.DetectorArray(
        positions, np.tile([0., 0., -1.], (len(positions), 1)), 0.15,
        lambda c: 0.3 + 0.7 * np.clip((np.asarray(c) + 1) / 2, 0, 1),
        lambda w: 0.2 + 0.1 * np.sin(np.asarray(w, float) / 40.),
        receiver_surface=lh.SphereSurface(0.216),
        identifiers={"string_id": np.repeat(np.arange(5), len(positions) // 5)})
    return lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=6, threshold_pe=0.,
        relative_time_edges_ns=np.arange(-20., 300., 5.)))
'''
namespace = {}
exec(KERNEL, namespace)
kernel = namespace["kernel"]


def compare(got, ref, rtol=1e-11):
    for key in ("steps", "ring_steps", "pairs", "tested"):
        np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
    for key in ("light", "ring_light", "charge", "skipped", "earliest_ns", "bins"):
        a, b = np.asarray(got[key]), np.asarray(ref[key])
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        finite = np.isfinite(b)
        scale = np.max(np.abs(b[finite]), initial=0.0)
        np.testing.assert_allclose(a[finite], b[finite], rtol=rtol, atol=rtol * scale, err_msg=key)
    for k in range(len(ref["cell_key"])):
        ka, kb = got["cell_key"][k], ref["cell_key"][k]
        oa, ob = np.lexsort(ka.T[::-1]), np.lexsort(kb.T[::-1])
        np.testing.assert_array_equal(ka[oa], kb[ob])
        np.testing.assert_allclose(got["cell_sums"][k][oa], ref["cell_sums"][k][ob], rtol=1e-12, atol=1e-300)


def test_sink_equals_lighthit_and_does_not_depend_on_threads_or_blocks():
    from lighthit_g4 import LightSink
    k = kernel()
    shower = lh.SyntheticShower.gaussian(lh.SourcePose([3., -2., 5.], [0.3, 0.2, 0.93], 4.0),
                                         charged_track_length_m=30., elements=3000, seed=2)
    steps = replace(shower.elements, cone_model="spectral")
    component = (np.arange(len(steps.length_m)) % 3 == 0).astype(np.int64)
    plan = StreamPlan.of(k, steps.start_m[0], times_ns=4.0, axis=(0.3, 0.2, 0.93), components=2)
    ref = stream_elements(plan, steps, component)
    results = []
    for threads, block in ((1, None), (3, 700)):
        sink = LightSink(plan, threads=threads, block=block)
        sink.add_steps(steps, component)
        sink.finish()
        results.append(sink.result())
        compare(results[-1], ref)
    np.testing.assert_array_equal(results[0]["charge"], filled_again(plan, steps, component)["charge"])
    assert results[0]["charge"].sum() > 0 and len(results[0]["cell_key"][1]) > 0


def filled_again(plan, steps, component):
    from lighthit_g4 import LightSink
    sink = LightSink(plan, threads=2)
    sink.add_steps(steps, component)
    sink.finish()
    return sink.result()


GEANT4 = KERNEL + '''
import sys
from dataclasses import replace
import numpy as np
from lighthit.sources import G4Shower
from lighthit.streamed import StreamPlan, stream_elements
from lighthit_g4 import Config, LightSink, Primary, Simulation, Writer

out = sys.argv[1]
vertex = (2.0, -3.0, 20.0)
direction = (0.3, 0.2, -0.93)
sim = Simulation(Config(center_m=vertex, water_radius_m=200.0, water_height_m=300.0))
plan = StreamPlan.of(kernel(), vertex, axis=direction, components=3)
sink = LightSink(plan, threads=3, block=5000)
primaries = [Primary(211, 3.0, vertex, direction), Primary(15, 50.0, vertex, direction)]
event = sim.run(primaries, seed=4, light=sink, components=[0, 1], decay_components=[None, 2])
with Writer(out, sim) as w:
    w.write(event)
np.savez(out + ".npz", **{k: v for k, v in event.light.items() if not k.startswith("cell_")},
         **{f"cell_key_{k}": v for k, v in enumerate(event.light["cell_key"])},
         **{f"cell_sums_{k}": v for k, v in enumerate(event.light["cell_sums"])})
lean = sim.run(primaries, seed=4, light=sink, components=[0, 1], decay_components=[None, 2],
               tracks=False, particles="primaries")
np.save(out + ".lean.npy", lean.light["charge"])
print(len(event.tracks), len(lean.tracks), len(event.particles), len(lean.particles))
'''


def test_geant4_steps_into_the_sink_equal_lighthit_on_the_written_tracks(tmp_path):
    out = tmp_path / "event.h5"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    run = subprocess.run([sys.executable, "-c", GEANT4, str(out)], env=env, capture_output=True,
                         text=True)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    rows, lean_rows, listed, lean_listed = map(int, run.stdout.split()[-4:])
    assert lean_rows == 0 and 0 < lean_listed < listed
    from lighthit.sources import G4Shower
    from lighthit_g4 import read_event
    saved = np.load(str(out) + ".npz")
    got = {k: saved[k] for k in saved.files if not k.startswith("cell_")}
    got["cell_key"] = [saved[f"cell_key_{k}"] for k in range(3)]
    got["cell_sums"] = [saved[f"cell_sums_{k}"] for k in range(3)]
    event = read_event(out, 0)
    parts = event.particles
    uid, parent = parts["uid"], parts["parent_uid"]
    group = np.full(len(parts), -1)
    group[uid == 1] = 0
    group[uid == 2] = 1
    group[(parent == 2) & (parts["creator"] == b"Decay")] = 2
    index = np.searchsorted(uid, parent)
    while np.any(group < 0):
        unknown = np.flatnonzero(group < 0)
        group[unknown] = group[index[unknown]]
    steps = replace(G4Shower.from_hdf5(out, 0).elements, cone_model="spectral")
    vertex, direction = (2.0, -3.0, 20.0), (0.3, 0.2, -0.93)
    plan = StreamPlan.of(kernel(), vertex, axis=direction, components=3)
    ref = stream_elements(plan, steps, group[np.searchsorted(uid, steps.uid)])
    compare(got, ref, rtol=1e-10)
    assert np.all(got["steps"] > 0)
    # the same event without keeping steps or the particle table: the same light
    np.testing.assert_array_equal(np.load(str(out) + ".lean.npy"), got["charge"])
