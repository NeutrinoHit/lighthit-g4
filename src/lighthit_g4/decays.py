"""Decays found in recorded events."""
import numpy as np

__all__ = ["decay_products", "decay_vertex", "tau_decay_mode"]


def decay_products(event, uid=1):
    """Particle rows made by the ``Decay`` process of particle ``uid``
    (empty when it did not decay in the event)."""
    daughters = event.daughters(uid)
    return daughters[daughters["creator"] == b"Decay"]


def decay_vertex(event, uid=1):
    """Position (m) and time (ns) of the decay of particle ``uid``, or None."""
    decayed = decay_products(event, uid)
    if not len(decayed):
        return None
    row = decayed[0]
    return np.array([row["x_m"], row["y_m"], row["z_m"]]), float(row["t_ns"])


def tau_decay_mode(event, uid=1):
    """Example classifier for a tau: ``"e"``, ``"mu"`` or ``"hadrons"``, or None
    when the tau did not decay in the event."""
    decayed = decay_products(event, uid)
    if not len(decayed):
        return None
    charged = {int(abs(p)) for p in decayed["pdgid"] if abs(p) not in (12, 14, 16, 22)}
    if charged == {11}:
        return "e"
    if charged == {13}:
        return "mu"
    return "hadrons"
