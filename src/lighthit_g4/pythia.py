"""Tau decays by Pythia 8 for the C++ core (``Config(tau_decayer="pythia")``).

Pythia comes from its own package ``pythia8mc`` (``pip install
'lighthit-g4[pythia]'``); it is not part of the lighthit-g4 wheel.  Geant4
calls :class:`PythiaDecays` once per tau decay with the tau four-momentum in
the laboratory frame, its helicity and a seed drawn from the Geant4 engine.

Pythia takes the helicity of an external tau as a pure state, -1 or +1 (the
SPINUP convention, ``TauDecays:mode = 1``; see "Hadron & Tau Decays" in the
Pythia manual), so each decay draws +1 with probability (1 + h)/2 and -1
otherwise: for a longitudinal polarization h this mixture is the same
density matrix.  Pythia decays the tau and short-lived resonances; every
product that Geant4 tracks itself (pi0, eta, kaons, ...) is returned
undecayed and decayed by Geant4.
"""
import os
import sys
from importlib import metadata
from pathlib import Path

__all__ = ["PythiaDecays"]


def _xmldoc(module):
    if os.environ.get("PYTHIA8DATA"):
        return Path(os.environ["PYTHIA8DATA"])
    here = Path(module.__file__).resolve().parent
    for base in (here, Path(sys.prefix), Path(sys.base_prefix)):
        candidate = base / "share" / "Pythia8" / "xmldoc"
        if (candidate / "Index.xml").is_file():
            return candidate
    raise FileNotFoundError("Pythia 8 xmldoc directory not found; set PYTHIA8DATA")


class PythiaDecays:
    def __init__(self, verbose=0):
        try:
            import pythia8mc
        except ImportError as exc:
            raise ImportError("tau_decayer='pythia' needs the pythia8mc package: "
                              "pip install 'lighthit-g4[pythia]'") from exc
        os.environ.setdefault("PYTHIA8DATA", str(_xmldoc(pythia8mc)))
        self.pythia = pythia8mc.Pythia("", verbose > 0)
        for line in ("ProcessLevel:all = off", "Next:numberCount = 0",
                     "Next:numberShowEvent = 0", "Next:numberShowInfo = 0",
                     "Next:numberShowProcess = 0", "Init:showChangedSettings = off",
                     "Init:showChangedParticleData = off"):
            self.pythia.readString(line)
        if verbose == 0:
            self.pythia.readString("Print:quiet = on")
        if not self.pythia.init():
            raise RuntimeError("Pythia initialization failed")
        self.description = f"Pythia {self.pythia.parm('Pythia:versionNumber'):.3f} (pythia8mc {metadata.version('pythia8mc')})"

    def leave_to_geant4(self, pdgids):
        """Switch off Pythia decays of the particles Geant4 tracks itself."""
        data = self.pythia.particleData
        for code in pdgids:
            if abs(code) != 15 and data.isParticle(code):
                data.mayDecay(code, False)

    def __call__(self, pdgid, px, py, pz, e, mass, helicity, seed):
        pythia = self.pythia
        pythia.rndm.init(int(seed))
        event = pythia.event
        event.reset()
        event.append(int(pdgid), 1, 0, 0, px, py, pz, e, mass)
        first = event.size()
        event[first - 1].pol(1.0 if pythia.rndm.flat() < 0.5 * (1.0 + helicity) else -1.0)
        if not pythia.next():
            raise RuntimeError(f"Pythia could not decay PDG {pdgid}")
        return [(p.id(), p.px(), p.py(), p.pz())
                for p in (event[i] for i in range(first, event.size())) if p.isFinal()]
