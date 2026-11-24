// lighthit_g4._core: the Geant4 application in C++.
//
// The same application as the geant4_pybind one in lighthit_g4/_g4py.py
// (water cylinder, particle gun, light-threshold kill rule, step recording),
// plus external decays of chosen particles through a Python function,
// with the user actions and the recording in C++.  Configuration and the
// file format stay in Python; rows are returned as raw bytes laid out as
// TRACK_DTYPE and PARTICLE_DTYPE.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "G4DecayTable.hh"
#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProductionCutsTable.hh"
#include "G4RunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4Tubs.hh"
#include "G4UIsession.hh"
#include "G4UImanager.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VDecayChannel.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VProcess.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "Randomize.hh"

#include "external_decayer.hh"

namespace nb = nanobind;
using namespace nb::literals;

namespace lhg4 {

#pragma pack(push, 1)
struct TrackRow {
  std::int64_t uid, parent_uid, pdgid;
  double x_m, y_m, z_m, t_ns, beta, step_length_m, e_kin_GeV;
};
struct ParticleRow {
  std::int64_t uid, parent_uid, pdgid;
  double mass_GeV;
  char creator[24];
  double x_m, y_m, z_m, t_ns, dx, dy, dz, e_kin_GeV;
  bool tracked;
};
#pragma pack(pop)
static_assert(sizeof(TrackRow) == 80, "TRACK_DTYPE layout");
static_assert(sizeof(ParticleRow) == 121, "PARTICLE_DTYPE layout");

struct Settings {
  std::string physics_list;
  double water_radius_m, water_height_m;
  std::array<double, 3> center_m;
  double production_cut_m, cherenkov_index;
  bool kill_below_threshold, track_secondaries;
  int verbose;
};

struct Shot {
  int pdgid = 0;
  double energy_GeV = 0, time_ns = 0, polarization = 0;
  std::array<double, 3> position_m{}, direction{0, 0, 1};
};

// Everything the actions of one event share.
struct State {
  Settings settings;
  double threshold_e_MeV = 0;
  Shot shot;
  std::vector<TrackRow> tracks;
  std::vector<ParticleRow> particles;
  long below_threshold = 0;
  // current track
  std::int64_t uid = 0, parent = 0, pdg = 0;
  bool charged = false;
  double electron_cut = -1;   // internal energy units; < 0 means no in-flight kill

  void reset()
  {
    tracks.clear();
    particles.clear();
    below_threshold = 0;
    uid = parent = pdg = 0;
    charged = false;
    electron_cut = -1;
  }

  void new_track(const G4Track* track, bool tracked)
  {
    const auto& c = settings.center_m;
    const G4ThreeVector position = track->GetPosition();
    const G4ThreeVector direction = track->GetMomentumDirection();
    const G4VProcess* process = track->GetCreatorProcess();
    const G4ParticleDefinition* definition = track->GetDefinition();
    ParticleRow row{};
    row.uid = track->GetTrackID();
    row.parent_uid = track->GetParentID();
    row.pdgid = definition->GetPDGEncoding();
    row.mass_GeV = definition->GetPDGMass() / GeV;
    const std::string creator = process == nullptr ? "primary" : process->GetProcessName();
    std::strncpy(row.creator, creator.c_str(), sizeof(row.creator));
    row.x_m = position.x() / m + c[0];
    row.y_m = position.y() / m + c[1];
    row.z_m = position.z() / m + c[2];
    row.t_ns = track->GetGlobalTime() / ns;
    row.dx = direction.x();
    row.dy = direction.y();
    row.dz = direction.z();
    row.e_kin_GeV = track->GetKineticEnergy() / GeV;
    row.tracked = tracked;
    particles.push_back(row);
  }

  void point(const G4StepPoint* point, double step_length)
  {
    const auto& c = settings.center_m;
    const G4ThreeVector position = point->GetPosition();
    tracks.push_back(TrackRow{uid, parent, pdg, position.x() / m + c[0], position.y() / m + c[1],
                              position.z() / m + c[2], point->GetGlobalTime() / ns,
                              point->GetBeta(), step_length / m,
                              point->GetKineticEnergy() / GeV});
  }

  // stacking rule for a secondary: 1 track, 0 list only, -1 kill below threshold
  int keep_new(const G4Track* track) const
  {
    const int code = track->GetDefinition()->GetPDGEncoding();
    const int a = std::abs(code);
    if (!settings.track_secondaries || a == 12 || a == 14 || a == 16) return 0;
    if (!settings.kill_below_threshold) return 1;
    if ((code == 11 || code == 22) && track->GetKineticEnergy() / MeV < threshold_e_MeV) return -1;
    return 1;
  }
};

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Settings& s) : s_(s) {}
  G4VPhysicalVolume* Construct() override
  {
    auto* water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
    auto* solid = new G4Tubs("Water", 0.0, s_.water_radius_m * m, 0.5 * s_.water_height_m * m,
                             0.0, 2 * M_PI);
    auto* logical = new G4LogicalVolume(solid, water, "Water");
    return new G4PVPlacement(nullptr, G4ThreeVector(), logical, "Water", nullptr, false, 0, true);
  }

 private:
  Settings s_;
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Gun(State& state) : state_(state), gun_(1) {}
  void GeneratePrimaries(G4Event* event) override
  {
    const Shot& shot = state_.shot;
    const auto& c = state_.settings.center_m;
    G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle(shot.pdgid);
    const G4ThreeVector direction(shot.direction[0], shot.direction[1], shot.direction[2]);
    gun_.SetParticleDefinition(definition);
    gun_.SetParticleEnergy(shot.energy_GeV * GeV);
    gun_.SetParticlePosition(G4ThreeVector((shot.position_m[0] - c[0]) * m,
                                           (shot.position_m[1] - c[1]) * m,
                                           (shot.position_m[2] - c[2]) * m));
    gun_.SetParticleMomentumDirection(direction);
    gun_.SetParticleTime(shot.time_ns * ns);
    gun_.SetParticlePolarization(shot.polarization * direction);
    gun_.GeneratePrimaryVertex(event);
  }

 private:
  State& state_;
  G4ParticleGun gun_;
};

class Stacking : public G4UserStackingAction {
 public:
  explicit Stacking(State& state) : state_(state) {}
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override
  {
    if (track->GetParentID() == 0) {
      state_.new_track(track, true);
      return fUrgent;
    }
    const int keep = state_.keep_new(track);
    if (keep < 0) {
      ++state_.below_threshold;
      return fKill;
    }
    state_.new_track(track, keep == 1);
    return keep == 1 ? fUrgent : fKill;
  }

 private:
  State& state_;
};

class Tracking : public G4UserTrackingAction {
 public:
  explicit Tracking(State& state) : state_(state) {}
  void PreUserTrackingAction(const G4Track* track) override
  {
    const G4ParticleDefinition* definition = track->GetDefinition();
    state_.uid = track->GetTrackID();
    state_.parent = track->GetParentID();
    state_.pdg = definition->GetPDGEncoding();
    state_.charged = definition->GetPDGCharge() != 0;
    state_.electron_cut = (state_.pdg == 11 && state_.settings.kill_below_threshold)
                            ? state_.threshold_e_MeV * MeV : -1.0;
  }

 private:
  State& state_;
};

class Stepping : public G4UserSteppingAction {
 public:
  explicit Stepping(State& state) : state_(state) {}
  void UserSteppingAction(const G4Step* step) override
  {
    if (!state_.charged) return;
    G4Track* track = step->GetTrack();
    if (track->GetCurrentStepNumber() == 1) state_.point(step->GetPreStepPoint(), 0.0);
    const G4StepPoint* post = step->GetPostStepPoint();
    const double length = step->GetStepLength();
    // an at-rest step (annihilation, decay) repeats the last vertex
    if (length > 0) state_.point(post, length);
    if (state_.electron_cut >= 0 && post->GetKineticEnergy() < state_.electron_cut)
      track->SetTrackStatus(fStopAndKill);
  }

 private:
  State& state_;
};

class Actions : public G4VUserActionInitialization {
 public:
  explicit Actions(State& state) : state_(state) {}
  void Build() const override
  {
    SetUserAction(new Gun(state_));
    SetUserAction(new Stacking(state_));
    SetUserAction(new Tracking(state_));
    SetUserAction(new Stepping(state_));
  }

 private:
  State& state_;
};

// G4cout goes nowhere unless verbose; G4cerr stays on stderr.
class Quiet : public G4UIsession {
 public:
  G4UIsession* SessionStart() override { return nullptr; }
  G4int ReceiveG4cout(const G4String&) override { return 0; }
  G4int ReceiveG4cerr(const G4String& text) override
  {
    std::fputs(text.c_str(), stderr);
    return 0;
  }
};

double kinetic_threshold_MeV(double mass_MeV, double index)
{
  if (index <= 1) throw std::invalid_argument("the Cherenkov index must exceed 1");
  return mass_MeV * (1.0 / std::sqrt(1.0 - 1.0 / std::pow(index, 2)) - 1.0);
}

class Engine {
 public:
  Engine(const Settings& settings, nb::object decayer, std::vector<int> external_decays)
  {
    if (instances_++ > 0) throw std::runtime_error("Geant4 allows one Simulation per process");
    state_.settings = settings;
    if (settings.verbose == 0) {
      quiet_ = std::make_unique<Quiet>();
      G4UImanager::GetUIpointer()->SetCoutDestination(quiet_.get());
    }
    run_manager_.reset(G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial));
    G4PhysListFactory factory(settings.verbose);
    if (!factory.IsReferencePhysList(settings.physics_list))
      throw std::invalid_argument(settings.physics_list + " is not a Geant4 reference physics list");
    G4VModularPhysicsList* physics = factory.GetReferencePhysList(settings.physics_list);
    physics->SetVerboseLevel(settings.verbose);
    if (!external_decays.empty()) {
      if (decayer.is_none()) throw std::invalid_argument("external decays need a decay function");
      external_ = new ExternalDecayPhysics(decayer, external_decays);
      physics->RegisterPhysics(external_);
      external_pdgids_ = external_decays;
    }
    run_manager_->SetUserInitialization(physics);
    electron_mass_MeV = G4Electron::Definition()->GetPDGMass() / MeV;
    state_.threshold_e_MeV = kinetic_threshold_MeV(electron_mass_MeV, settings.cherenkov_index);
    run_manager_->SetUserInitialization(new Detector(settings));
    run_manager_->SetUserInitialization(new Actions(state_));
    auto* ui = G4UImanager::GetUIpointer();
    for (const char* command : {"/control/verbose", "/run/verbose", "/event/verbose",
                                "/tracking/verbose", "/process/em/verbose", "/process/had/verbose"})
      ui->ApplyCommand(std::string(command) + " " + std::to_string(settings.verbose));
    std::ostringstream cut;
    cut.precision(17);
    cut << "/run/setCut " << settings.production_cut_m * 1e3 << " mm";
    ui->ApplyCommand(cut.str());
    run_manager_->Initialize();
  }

  ~Engine()
  {
    run_manager_.reset();
    if (quiet_) G4UImanager::GetUIpointer()->SetCoutDestination(nullptr);
  }

  nb::tuple run(const Shot& shot, long seed)
  {
    if (G4ParticleTable::GetParticleTable()->FindParticle(shot.pdgid) == nullptr)
      throw std::invalid_argument("Geant4 does not know PDG " + std::to_string(shot.pdgid));
    state_.shot = shot;
    state_.reset();
    G4Random::setTheSeed(seed);
    const std::clock_t start = std::clock();
    {
      nb::gil_scoped_release release;
      run_manager_->BeamOn(1);
    }
    const double cpu = double(std::clock() - start) / CLOCKS_PER_SEC;
    if (external_ != nullptr && external_->decayer != nullptr && !external_->decayer->error.empty()) {
      const std::string error = external_->decayer->error;
      external_->decayer->error.clear();
      throw std::runtime_error("external decay failed, event aborted: " + error);
    }
    auto by_uid = [](const auto& a, const auto& b) { return a.uid < b.uid; };
    std::stable_sort(state_.tracks.begin(), state_.tracks.end(), by_uid);
    std::stable_sort(state_.particles.begin(), state_.particles.end(), by_uid);
    nb::bytes tracks(reinterpret_cast<const char*>(state_.tracks.data()),
                     state_.tracks.size() * sizeof(TrackRow));
    nb::bytes particles(reinterpret_cast<const char*>(state_.particles.data()),
                        state_.particles.size() * sizeof(ParticleRow));
    return nb::make_tuple(tracks, particles, state_.below_threshold, cpu);
  }

  // valid once an event has built the cuts table
  nb::dict energy_cuts_MeV() const
  {
    auto* water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
    auto* table = G4ProductionCutsTable::GetProductionCutsTable();
    nb::dict out;
    for (const char* name : {"gamma", "e-", "e+", "proton"}) {
      G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle(name);
      out[name] = table->ConvertRangeToEnergy(definition, water,
                                              state_.settings.production_cut_m * m) / MeV;
    }
    return out;
  }

  nb::dict particle_properties(int pdgid) const
  {
    G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle(pdgid);
    if (definition == nullptr)
      throw std::invalid_argument("Geant4 does not know PDG " + std::to_string(pdgid));
    nb::dict out;
    out["name"] = std::string(definition->GetParticleName());
    out["mass_MeV"] = definition->GetPDGMass() / MeV;
    out["charge"] = definition->GetPDGCharge() / eplus;
    out["stable"] = bool(definition->GetPDGStable());
    out["lifetime_ns"] = definition->GetPDGLifeTime() / ns;
    nb::list channels;
    if (G4DecayTable* table = definition->GetDecayTable()) {
      for (G4int i = 0; i < table->entries(); ++i) {
        G4VDecayChannel* channel = table->GetDecayChannel(i);
        nb::list daughters;
        for (G4int j = 0; j < channel->GetNumberOfDaughters(); ++j)
          daughters.append(std::string(channel->GetDaughterName(j)));
        nb::dict entry;
        entry["br"] = channel->GetBR();
        entry["kinematics"] = std::string(channel->GetKinematicsName());
        entry["daughters"] = daughters;
        channels.append(entry);
      }
    }
    out["decay_table"] = channels;
    const bool external = std::find(external_pdgids_.begin(), external_pdgids_.end(), pdgid)
                          != external_pdgids_.end();
    out["decayer"] = external ? "external" : "Geant4";
    return out;
  }

  nb::dict decayer_counts() const
  {
    nb::dict out;
    if (external_ != nullptr && external_->decayer != nullptr) {
      out["decays"] = external_->decayer->decays;
      out["unknown_products"] = external_->decayer->unknown_products;
    }
    return out;
  }

  // PDG codes of the particles Geant4 tracks itself (not short-lived resonances)
  std::vector<int> tracked_species() const
  {
    std::vector<int> out;
    auto* iterator = G4ParticleTable::GetParticleTable()->GetIterator();
    iterator->reset();
    while ((*iterator)()) {
      G4ParticleDefinition* particle = iterator->value();
      if (!particle->IsShortLived() && particle->GetPDGEncoding() != 0)
        out.push_back(particle->GetPDGEncoding());
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  double electron_mass_MeV = 0;
  double threshold_e_MeV() const { return state_.threshold_e_MeV; }

 private:
  static inline int instances_ = 0;
  State state_;
  std::unique_ptr<Quiet> quiet_;
  std::unique_ptr<G4RunManager> run_manager_;
  ExternalDecayPhysics* external_ = nullptr;   // owned by the physics list
  std::vector<int> external_pdgids_;
};

}  // namespace lhg4

NB_MODULE(_core, module)
{
  using namespace lhg4;
  nb::set_leak_warnings(false);
  module.attr("geant4_version") = std::string(G4Version);

  nb::class_<Settings>(module, "Settings")
    .def(nb::init<>())
    .def_rw("physics_list", &Settings::physics_list)
    .def_rw("water_radius_m", &Settings::water_radius_m)
    .def_rw("water_height_m", &Settings::water_height_m)
    .def_rw("center_m", &Settings::center_m)
    .def_rw("production_cut_m", &Settings::production_cut_m)
    .def_rw("cherenkov_index", &Settings::cherenkov_index)
    .def_rw("kill_below_threshold", &Settings::kill_below_threshold)
    .def_rw("track_secondaries", &Settings::track_secondaries)
    .def_rw("verbose", &Settings::verbose);

  nb::class_<Shot>(module, "Shot")
    .def(nb::init<>())
    .def_rw("pdgid", &Shot::pdgid)
    .def_rw("energy_GeV", &Shot::energy_GeV)
    .def_rw("position_m", &Shot::position_m)
    .def_rw("direction", &Shot::direction)
    .def_rw("time_ns", &Shot::time_ns)
    .def_rw("polarization", &Shot::polarization);

  nb::class_<Engine>(module, "Engine")
    .def(nb::init<const Settings&, nb::object, std::vector<int>>(), "settings"_a,
         "decayer"_a = nb::none(), "external_decays"_a = std::vector<int>())
    .def("run", &Engine::run, "shot"_a, "seed"_a)
    .def("energy_cuts_MeV", &Engine::energy_cuts_MeV)
    .def("particle_properties", &Engine::particle_properties, "pdgid"_a)
    .def("decayer_counts", &Engine::decayer_counts)
    .def("tracked_species", &Engine::tracked_species)
    .def_ro("electron_mass_MeV", &Engine::electron_mass_MeV)
    .def_prop_ro("threshold_e_MeV", &Engine::threshold_e_MeV);
}
