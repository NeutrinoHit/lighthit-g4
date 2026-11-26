"""The light of the steps without keeping the steps (LightHit's streamed showers).

A :class:`LightSink` is filled while Geant4 runs (``Simulation.run(...,
light=sink)``): per light component and per OM the direct light, the earliest
arrival and the sums of the finest emission cells, as
``lighthit.streamed.stream_elements`` computes them from stored steps. The
plan (OM tables, clock, cell frame) is LightHit's: build it with
``lighthit.streamed.StreamPlan.of(kernel, vertices, ...)`` and turn the result
into sources with ``lighthit.streamed.streamed_showers(plan, event.light)``.
"""
import numpy as np

__all__ = ["LightSink", "PARTICLES"]

#: what ``Simulation.run(particles=...)`` keeps of the particle table
PARTICLES = {"all": 0, "primaries": 1, "none": 2}


class LightSink:
    """C++ light sink for one plan; ``threads`` workers (default: all cores)."""

    def __init__(self, plan, *, threads=None, block=None):
        from . import _core
        arrays = getattr(plan, "arrays", plan)
        self._sink = _core.LightSink(dict(arrays), int(threads or 0), int(block or 0))
        self.plan = plan

    @property
    def components(self):
        return self._sink.components

    @property
    def threads(self):
        return self._sink.threads

    def reset(self):
        self._sink.reset()

    def finish(self):
        self._sink.finish()

    def add_steps(self, elements, component=None):
        """Stored steps (``lighthit.SpectralLightElements``, threshold already applied)."""
        n = len(elements.length_m)
        component = np.zeros(n, np.int64) if component is None else np.asarray(component, np.int64)
        c = np.ascontiguousarray
        self._sink.add_steps(c(elements.start_m, float), c(elements.direction, float),
                             c(elements.length_m, float), c(elements.coefficient0, float),
                             c(elements.coefficient2, float), c(elements.beta, float),
                             c(elements.start_ns, float), c(elements.end_ns, float), c(component))

    def add_rows(self, tracks, component=None):
        """Stored vertices in the track format (rows of one ``uid`` contiguous)."""
        n = len(tracks)
        component = np.zeros(n, np.int64) if component is None else np.asarray(component, np.int64)
        c = np.ascontiguousarray
        self._sink.add_points(c(tracks["uid"], np.int64), c(component), c(tracks["x_m"], float),
                              c(tracks["y_m"], float), c(tracks["z_m"], float),
                              c(tracks["t_ns"], float), c(tracks["beta"], float),
                              c(tracks["step_length_m"], float))

    def result(self):
        """Dictionary of arrays (the layout of ``lighthit.streamed.stream_elements``)."""
        return self._sink.result()
