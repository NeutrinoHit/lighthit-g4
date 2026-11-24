"""The same Geant4 application through geant4_pybind, with the user actions in
Python.  Kept as a reference for the C++ core (``Config(backend="geant4_pybind")``):
both apply the same rules in the same order, so the same seed gives the same event.
"""
import math
import time

import numpy as np

import geant4_pybind as g4

from .simulation import NEUTRINOS, PARTICLE_DTYPE, TRACK_DTYPE, kinetic_threshold_MeV

class _Recorder:
    """Python-side buffers filled by the user actions of one event."""

    def __init__(self, sim):
        self.sim = sim
        self.offset = np.asarray(sim.config.center_m, float)
        self.reset()

    def reset(self):
        self.tracks = []
        self.particles = []
        self.below_threshold = 0
        self.uid = self.parent = self.pdg = 0
        self.charged = False
        self.electron_cut = None

    # one row per new track (stacking action)
    def new_track(self, track, tracked):
        position = track.GetPosition()
        direction = track.GetMomentumDirection()
        process = track.GetCreatorProcess()
        creator = "primary" if process is None else process.GetProcessName()
        definition = track.GetDefinition()
        self.particles.append((
            track.GetTrackID(), track.GetParentID(), definition.GetPDGEncoding(),
            definition.GetPDGMass() / g4.GeV, creator.encode()[:24],
            position.x / g4.m + self.offset[0], position.y / g4.m + self.offset[1],
            position.z / g4.m + self.offset[2], track.GetGlobalTime() / g4.ns,
            direction.x, direction.y, direction.z,
            track.GetKineticEnergy() / g4.GeV, tracked))

    def point(self, point, step_length):
        position = point.GetPosition()
        self.tracks.append((
            self.uid, self.parent, self.pdg,
            position.x / g4.m + self.offset[0], position.y / g4.m + self.offset[1],
            position.z / g4.m + self.offset[2], point.GetGlobalTime() / g4.ns,
            point.GetBeta(), step_length / g4.m, point.GetKineticEnergy() / g4.GeV))

    def finish(self):
        tracks = np.array(self.tracks, TRACK_DTYPE) if self.tracks else np.empty(0, TRACK_DTYPE)
        # one block per particle, steps in Geant4 order inside the block
        tracks = tracks[np.argsort(tracks["uid"], kind="stable")]
        particles = (np.array(self.particles, PARTICLE_DTYPE) if self.particles
                     else np.empty(0, PARTICLE_DTYPE))
        particles = particles[np.argsort(particles["uid"], kind="stable")]
        return tracks, particles


class _Detector(g4.G4VUserDetectorConstruction):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.keep = []

    def Construct(self):
        water = g4.G4NistManager.Instance().FindOrBuildMaterial("G4_WATER")
        solid = g4.G4Tubs("Water", 0.0, self.config.water_radius_m * g4.m,
                          0.5 * self.config.water_height_m * g4.m, 0.0, 2 * math.pi)
        logical = g4.G4LogicalVolume(solid, water, "Water")
        world = g4.G4PVPlacement(None, g4.G4ThreeVector(), logical, "Water",
                                 None, False, 0, True)
        self.keep += [solid, logical, world]
        return world


class _Gun(g4.G4VUserPrimaryGeneratorAction):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.gun = g4.G4ParticleGun(1)

    def GeneratePrimaries(self, event):
        # one primary vertex per particle; Geant4 numbers them 1..N in this order
        for primary in self.sim.current:
            definition = g4.G4ParticleTable.GetParticleTable().FindParticle(int(primary.pdgid))
            position = np.asarray(primary.position_m, float) - self.sim.recorder.offset
            polarization = primary.polarization * np.asarray(primary.direction, float)
            self.gun.SetParticleDefinition(definition)
            self.gun.SetParticleEnergy(primary.energy_GeV * g4.GeV)
            self.gun.SetParticlePosition(g4.G4ThreeVector(*(position * g4.m)))
            self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(*primary.direction))
            self.gun.SetParticleTime(primary.time_ns * g4.ns)
            self.gun.SetParticlePolarization(g4.G4ThreeVector(*polarization))
            self.gun.GeneratePrimaryVertex(event)


class _Stacking(g4.G4UserStackingAction):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.urgent = g4.G4ClassificationOfNewTrack.fUrgent
        self.kill = g4.G4ClassificationOfNewTrack.fKill

    def ClassifyNewTrack(self, track):
        recorder = self.sim.recorder
        if track.GetParentID() == 0:
            recorder.new_track(track, True)
            return self.urgent
        keep = self.sim.keep_new(track)
        if keep == "below":
            recorder.below_threshold += 1
            return self.kill
        recorder.new_track(track, keep)
        return self.urgent if keep else self.kill


class _Tracking(g4.G4UserTrackingAction):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim

    def PreUserTrackingAction(self, track):
        recorder = self.sim.recorder
        definition = track.GetDefinition()
        recorder.uid = track.GetTrackID()
        recorder.parent = track.GetParentID()
        recorder.pdg = definition.GetPDGEncoding()
        recorder.charged = definition.GetPDGCharge() != 0
        recorder.electron_cut = (self.sim.threshold_e_MeV * g4.MeV
                                 if recorder.pdg == 11 and self.sim.config.kill_below_threshold
                                 else None)


class _Stepping(g4.G4UserSteppingAction):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.stop = g4.G4TrackStatus.fStopAndKill

    def UserSteppingAction(self, step):
        recorder = self.sim.recorder
        if not recorder.charged:
            return
        track = step.GetTrack()
        if track.GetCurrentStepNumber() == 1:
            recorder.point(step.GetPreStepPoint(), 0.0)
        post = step.GetPostStepPoint()
        length = step.GetStepLength()
        if length > 0:      # an at-rest step (annihilation, decay) repeats the last vertex
            recorder.point(post, length)
        cut = recorder.electron_cut
        if cut is not None and post.GetKineticEnergy() < cut:
            track.SetTrackStatus(self.stop)


class _Actions(g4.G4VUserActionInitialization):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.keep = []

    def Build(self):
        actions = [_Gun(self.sim), _Stacking(self.sim), _Tracking(self.sim), _Stepping(self.sim)]
        for action in actions:
            self.SetUserAction(action)
        self.keep += actions



class PybindEngine:
    """Geant4 through geant4_pybind; the interface of the C++ engine."""

    def __init__(self, config, verbose=0):
        if config.tau_decayer != "geant4":
            raise ValueError("the geant4_pybind backend has only the Geant4 decay tables")
        self.config = config
        self.recorder = _Recorder(self)
        self.current = None
        self.run_manager = g4.G4RunManagerFactory.CreateRunManager(g4.G4RunManagerType.Serial)
        # built in Python: a list returned by G4PhysListFactory cannot be handed
        # back to the run manager through geant4_pybind
        physics_class = getattr(g4, config.physics_list, None)
        if physics_class is None or not issubclass(physics_class, g4.G4VModularPhysicsList):
            raise ValueError(f"{config.physics_list} is not a reference physics list of geant4_pybind")
        physics = physics_class()
        physics.SetVerboseLevel(verbose)
        # the run manager owns and deletes the list: a Python reference kept
        # beyond this call makes the interpreter delete it again at exit
        self.run_manager.SetUserInitialization(physics)
        del physics
        # particles may be touched only after the physics list is assigned
        self.electron_mass_MeV = g4.G4Electron.Definition().GetPDGMass() / g4.MeV
        self.threshold_e_MeV = kinetic_threshold_MeV(self.electron_mass_MeV, config.cherenkov_index)
        self.detector = _Detector(config)
        self.actions = _Actions(self)
        self.run_manager.SetUserInitialization(self.detector)
        self.run_manager.SetUserInitialization(self.actions)
        ui = g4.G4UImanager.GetUIpointer()
        for command in ("/control/verbose", "/run/verbose", "/event/verbose",
                        "/tracking/verbose", "/process/em/verbose", "/process/had/verbose"):
            ui.ApplyCommand(f"{command} {int(verbose)}")
        # through the UI: G4VUserPhysicsList::SetDefaultCutValue called from
        # Python makes the process crash when Geant4 is torn down at exit
        ui.ApplyCommand(f"/run/setCut {config.production_cut_m * 1e3!r} mm")
        self.run_manager.Initialize()

    geant4_version = g4.G4Version

    def keep_new(self, track):
        """Stacking rule for a secondary: True, False (listed, not tracked) or
        ``"below"`` (killed below the light threshold, not listed)."""
        pdg = track.GetDefinition().GetPDGEncoding()
        if self.config.secondaries == "none" or abs(pdg) in NEUTRINOS:
            return False
        if not self.config.kill_below_threshold:
            return True
        if pdg in (11, 22) and track.GetKineticEnergy() / g4.MeV < self.threshold_e_MeV:
            return "below"
        return True

    def run(self, primaries, seed):
        for primary in primaries:
            if g4.G4ParticleTable.GetParticleTable().FindParticle(int(primary.pdgid)) is None:
                raise ValueError(f"Geant4 does not know PDG {primary.pdgid}")
        g4.G4Random.setTheSeed(int(seed))
        self.current = tuple(primaries)
        self.recorder.reset()
        start = time.process_time()
        self.run_manager.BeamOn(1)
        cpu = time.process_time() - start
        tracks, particles = self.recorder.finish()
        return tracks, particles, self.recorder.below_threshold, cpu

    def energy_cuts_MeV(self):
        water = g4.G4NistManager.Instance().FindOrBuildMaterial("G4_WATER")
        table = g4.G4ProductionCutsTable.GetProductionCutsTable()
        out = {}
        for name in ("gamma", "e-", "e+", "proton"):
            definition = g4.G4ParticleTable.GetParticleTable().FindParticle(name)
            out[name] = table.ConvertRangeToEnergy(
                definition, water, self.config.production_cut_m * g4.m) / g4.MeV
        return out

    def particle_properties(self, pdgid):
        definition = g4.G4ParticleTable.GetParticleTable().FindParticle(int(pdgid))
        if definition is None:
            raise ValueError(f"Geant4 does not know PDG {pdgid}")
        name = definition.GetParticleName()
        return {"name": name, "mass_MeV": definition.GetPDGMass() / g4.MeV,
                "charge": definition.GetPDGCharge() / g4.eplus,
                "stable": bool(definition.GetPDGStable()),
                "lifetime_ns": definition.GetPDGLifeTime() / g4.ns,
                "decay_table": decay_table(name), "decayer": "Geant4"}

    def decayer_counts(self):
        return {}


def decay_table(name):
    """Channels of one particle: BR, kinematics model, daughters.

    geant4_pybind does not expose ``G4VDecayChannel``, so the table is read
    from ``G4DecayTable::DumpInfo``.
    """
    import contextlib
    import io
    import re
    definition = g4.G4ParticleTable.GetParticleTable().FindParticle(name)
    table = definition.GetDecayTable()
    if table is None:
        return []
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        table.DumpInfo()
    channels = []
    for line in text.getvalue().splitlines():
        match = re.match(r"\s*\d+:\s+BR:\s+(\S+)\s+\[(.+?)\]\s+:\s+(.*)", line)
        if match:
            channels.append({"br": float(match.group(1)), "kinematics": match.group(2),
                             "daughters": match.group(3).split()})
    if len(channels) != table.entries():
        raise RuntimeError("could not read every decay channel from DumpInfo")
    return channels
