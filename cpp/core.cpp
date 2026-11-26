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
#include <nanobind/ndarray.h>
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
#include "light_sink.hh"

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
  std::vector<Shot> shots;   // the primaries of the event, track IDs 1..N in this order
  std::vector<TrackRow> tracks;
  std::vector<ParticleRow> particles;
  long below_threshold = 0;
  // what is kept: the steps (tracks), the particles (0 all, 1 the primaries and
  // their daughters, 2 none) and the light of the steps (sink, may be null)
  bool record_tracks = true;
  int record_particles = 0;
  LightSink* sink = nullptr;
  // light components: of each primary, of the decay products of each primary
  // (-1: as the primary); every other particle has its parent's
  std::vector<int> primary_component, decay_component;
  std::vector<std::int8_t> component_of;   // by uid
  // current track
  std::int64_t uid = 0, parent = 0, pdg = 0;
  int component = 0;
  bool charged = false;
  double electron_cut = -1;   // internal energy units; < 0 means no in-flight kill

  void reset()
  {
    tracks.clear();
    particles.clear();
    component_of.assign(shots.size() + 1, 0);
    below_threshold = 0;
    uid = parent = pdg = 0;
    component = 0;
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
    const std::int64_t id = track->GetTrackID(), mother = track->GetParentID();
    const std::int64_t primaries = std::int64_t(shots.size());
    if (sink != nullptr) {
      int k = 0;
      if (mother == 0) k = id <= primaries ? primary_component[id - 1] : 0;
      else if (mother <= primaries && process != nullptr && process->GetProcessName() == "Decay"
               && decay_component[mother - 1] >= 0) k = decay_component[mother - 1];
      else k = mother < std::int64_t(component_of.size()) ? component_of[mother] : 0;
      if (id >= std::int64_t(component_of.size())) component_of.resize(std::size_t(id) * 3 / 2 + 16, 0);
      component_of[id] = std::int8_t(k);
    }
    if (record_particles == 2 || (record_particles == 1 && mother > primaries)) return;
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
    const double x = position.x() / m + c[0], y = position.y() / m + c[1], z = position.z() / m + c[2];
    const double t = point->GetGlobalTime() / ns, beta = point->GetBeta(), length = step_length / m;
    if (record_tracks)
      tracks.push_back(TrackRow{uid, parent, pdg, x, y, z, t, beta, length,
                                point->GetKineticEnergy() / GeV});
    if (sink != nullptr) sink->point(uid, component, x, y, z, t, beta, length);
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
  // one primary vertex per particle
  void GeneratePrimaries(G4Event* event) override
  {
    const auto& c = state_.settings.center_m;
    for (const Shot& shot : state_.shots) {
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
    state_.component = state_.uid < std::int64_t(state_.component_of.size())
                         ? state_.component_of[state_.uid] : 0;
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

  nb::tuple run(const std::vector<Shot>& shots, long seed, LightSink* sink,
                std::vector<int> components, std::vector<int> decay_components,
                bool record_tracks, int record_particles)
  {
    if (shots.empty()) throw std::invalid_argument("an event needs at least one primary");
    for (const Shot& shot : shots)
      if (G4ParticleTable::GetParticleTable()->FindParticle(shot.pdgid) == nullptr)
        throw std::invalid_argument("Geant4 does not know PDG " + std::to_string(shot.pdgid));
    if (record_particles < 0 || record_particles > 2)
      throw std::invalid_argument("record_particles is 0 (all), 1 (primaries and daughters) or 2 (none)");
    if (components.empty()) components.assign(shots.size(), 0);
    if (decay_components.empty()) decay_components.assign(shots.size(), -1);
    if (components.size() != shots.size() || decay_components.size() != shots.size())
      throw std::invalid_argument("one light component and one decay component per primary");
    if (sink != nullptr) {
      const int count = sink->plan().components;
      for (std::size_t i = 0; i < shots.size(); ++i)
        if (components[i] < 0 || components[i] >= count || decay_components[i] >= count
            || count > 127)
          throw std::invalid_argument("light components outside the sink plan");
    }
    state_.shots = shots;
    state_.sink = sink;
    state_.record_tracks = record_tracks;
    state_.record_particles = record_particles;
    state_.primary_component = components;
    state_.decay_component = decay_components;
    state_.reset();
    G4Random::setTheSeed(seed);
    const std::clock_t start = std::clock();
    {
      nb::gil_scoped_release release;
      if (sink != nullptr) sink->reset();
      run_manager_->BeamOn(1);
      if (sink != nullptr) sink->finish();
    }
    state_.sink = nullptr;
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

// ---- light sink: plan from the dictionary of lighthit.streamed.StreamPlan, results as arrays

using DoubleArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using IntArray = nb::ndarray<const std::int64_t, nb::c_contig, nb::device::cpu>;

std::vector<double> doubles(const nb::dict& d, const char* name)
{
  const DoubleArray a = nb::cast<DoubleArray>(d[name]);
  return std::vector<double>(a.data(), a.data() + a.size());
}

std::vector<std::int64_t> integers(const nb::dict& d, const char* name)
{
  const IntArray a = nb::cast<IntArray>(d[name]);
  return std::vector<std::int64_t>(a.data(), a.data() + a.size());
}

template <std::size_t N>
std::array<double, N> fixed(const nb::dict& d, const char* name)
{
  const std::vector<double> v = doubles(d, name);
  if (v.size() != N) throw std::invalid_argument(std::string("light sink plan: ") + name);
  std::array<double, N> out{};
  std::copy(v.begin(), v.end(), out.begin());
  return out;
}

SinkPlan sink_plan(const nb::dict& d)
{
  SinkPlan p;
  auto number = [&](const char* name) { return nb::cast<double>(d[name]); };
  auto count = [&](const char* name) { return nb::cast<int>(d[name]); };
  p.components = count("components");
  p.reference_phase_index = number("reference_phase_index");
  p.max_phase = number("max_phase");
  p.light_per_m = number("light_per_m");
  p.flight_ns_per_m = number("flight_ns_per_m");
  p.pos = doubles(d, "pos");
  p.ori = doubles(d, "ori");
  p.area = doubles(d, "area");
  p.R = number("R");
  p.n = doubles(d, "n");
  p.ng = doubles(d, "ng");
  p.mu = doubles(d, "mu");
  p.eg = doubles(d, "eg");
  p.lam0 = number("lam0");
  p.dlam = number("dlam");
  p.knots = doubles(d, "knots");
  p.acc = doubles(d, "acc");
  p.n_max = number("n_max");
  p.n_min = number("n_min");
  p.spec_max = number("spec_max");
  p.mu_min = number("mu_min");
  p.origins = doubles(d, "origins");
  p.edge0 = number("edge0");
  p.bin_w = number("bin_w");
  p.n_bins = count("n_bins");
  p.end_x = doubles(d, "end_x");
  p.end_w = doubles(d, "end_w");
  p.inner_x = doubles(d, "inner_x");
  p.inner_w = doubles(d, "inner_w");
  p.piece_m = number("piece_m");
  p.pair_tol = number("pair_tol");
  p.chunks = count("chunks");
  p.s_cx = doubles(d, "s_cx");
  p.s_cy = doubles(d, "s_cy");
  p.s_dev = doubles(d, "s_dev");
  p.s_start = integers(d, "s_start");
  p.s_count = integers(d, "s_count");
  p.om_order = integers(d, "om_order");
  p.om_z = doubles(d, "om_z");
  p.centre = fixed<3>(d, "frame_centre");
  p.axis = fixed<3>(d, "frame_axis");
  p.u = fixed<3>(d, "frame_u");
  p.v = fixed<3>(d, "frame_v");
  p.delay_origin = number("delay_origin");
  p.cell_size = fixed<5>(d, "cell_size");
  return p;
}

template <typename T>
nb::ndarray<nb::numpy, T> numpy_copy(const T* data, std::vector<std::size_t> shape)
{
  std::size_t size = 1;
  for (std::size_t n : shape) size *= n;
  T* copy = new T[size > 0 ? size : 1];
  std::copy(data, data + size, copy);
  nb::capsule owner(copy, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T>(copy, shape.size(), shape.data(), owner);
}

nb::dict sink_result(const LightSink& sink)
{
  const SinkPlan& p = sink.plan();
  const std::size_t K = p.components, n = p.n_om(), bins = p.n_bins;
  std::vector<double> charge, binned, skipped, earliest, light, ring_light;
  std::vector<std::int64_t> pairs, tested, steps, ring_steps;
  nb::list keys, sums;
  for (const ComponentResult& r : sink.results()) {
    charge.insert(charge.end(), r.charge.begin(), r.charge.end());
    binned.insert(binned.end(), r.bins.begin(), r.bins.end());
    skipped.insert(skipped.end(), r.skipped.begin(), r.skipped.end());
    earliest.insert(earliest.end(), r.earliest.begin(), r.earliest.end());
    light.push_back(r.light);
    ring_light.push_back(r.ring_light);
    pairs.push_back(r.pairs);
    tested.push_back(r.tested);
    steps.push_back(r.steps);
    ring_steps.push_back(r.ring_steps);
    const std::size_t cells = r.cells.size();
    keys.append(numpy_copy(r.cells.keys().data(), {cells, std::size_t(CELL_KEYS)}));
    sums.append(numpy_copy(r.cells.sums().data(), {cells, std::size_t(CELL_SUMS)}));
  }
  nb::dict out;
  out["charge"] = numpy_copy(charge.data(), {K, n});
  out["bins"] = numpy_copy(binned.data(), {K, n, bins});
  out["skipped"] = numpy_copy(skipped.data(), {K, n});
  out["earliest_ns"] = numpy_copy(earliest.data(), {K, n});
  out["pairs"] = numpy_copy(pairs.data(), {K});
  out["tested"] = numpy_copy(tested.data(), {K});
  out["steps"] = numpy_copy(steps.data(), {K});
  out["ring_steps"] = numpy_copy(ring_steps.data(), {K});
  out["light"] = numpy_copy(light.data(), {K});
  out["ring_light"] = numpy_copy(ring_light.data(), {K});
  out["cell_key"] = keys;
  out["cell_sums"] = sums;
  return out;
}

// stored steps or stored vertices into a sink (tests, files written earlier)
void sink_steps(LightSink& sink, DoubleArray start, DoubleArray direction, DoubleArray length,
                DoubleArray c0, DoubleArray c2, DoubleArray beta, DoubleArray t0, DoubleArray t1,
                IntArray component)
{
  const std::size_t n = length.size();
  if (start.size() != 3 * n || direction.size() != 3 * n || c0.size() != n || c2.size() != n
      || beta.size() != n || t0.size() != n || t1.size() != n || component.size() != n)
    throw std::invalid_argument("one row of every column per step");
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i)
    sink.add(SinkStep{start.data()[3 * i], start.data()[3 * i + 1], start.data()[3 * i + 2],
                      direction.data()[3 * i], direction.data()[3 * i + 1], direction.data()[3 * i + 2],
                      length.data()[i], c0.data()[i], c2.data()[i], beta.data()[i], t0.data()[i],
                      t1.data()[i], int(component.data()[i])});
}

void sink_points(LightSink& sink, IntArray uid, IntArray component, DoubleArray x, DoubleArray y,
                 DoubleArray z, DoubleArray t, DoubleArray beta, DoubleArray step_length)
{
  const std::size_t n = uid.size();
  if (component.size() != n || x.size() != n || y.size() != n || z.size() != n || t.size() != n
      || beta.size() != n || step_length.size() != n)
    throw std::invalid_argument("one row of every column per vertex");
  nb::gil_scoped_release release;
  for (std::size_t i = 0; i < n; ++i)
    sink.point(uid.data()[i], int(component.data()[i]), x.data()[i], y.data()[i], z.data()[i],
               t.data()[i], beta.data()[i], step_length.data()[i]);
}

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

  nb::class_<LightSink>(module, "LightSink")
    .def("__init__", [](LightSink* self, nb::dict plan, int threads, std::size_t block) {
           new (self) LightSink(sink_plan(plan), threads);
           if (block > 0) self->block_size = block;
         }, "plan"_a, "threads"_a = 0, "block"_a = 0)
    .def("reset", [](LightSink& s) { nb::gil_scoped_release r; s.reset(); })
    .def("finish", [](LightSink& s) { nb::gil_scoped_release r; s.finish(); })
    .def("add_steps", &sink_steps, "start"_a, "direction"_a, "length"_a, "c0"_a, "c2"_a,
         "beta"_a, "t0"_a, "t1"_a, "component"_a)
    .def("add_points", &sink_points, "uid"_a, "component"_a, "x"_a, "y"_a, "z"_a, "t"_a,
         "beta"_a, "step_length"_a)
    .def("result", &sink_result)
    .def_prop_ro("threads", &LightSink::threads)
    .def_prop_ro("components", [](const LightSink& s) { return s.plan().components; });

  nb::class_<Engine>(module, "Engine")
    .def(nb::init<const Settings&, nb::object, std::vector<int>>(), "settings"_a,
         "decayer"_a = nb::none(), "external_decays"_a = std::vector<int>())
    .def("run", &Engine::run, "shots"_a, "seed"_a, "sink"_a.none() = nb::none(),
         "components"_a = std::vector<int>(), "decay_components"_a = std::vector<int>(),
         "record_tracks"_a = true, "record_particles"_a = 0)
    .def("energy_cuts_MeV", &Engine::energy_cuts_MeV)
    .def("particle_properties", &Engine::particle_properties, "pdgid"_a)
    .def("decayer_counts", &Engine::decayer_counts)
    .def("tracked_species", &Engine::tracked_species)
    .def_ro("electron_mass_MeV", &Engine::electron_mass_MeV)
    .def_prop_ro("threshold_e_MeV", &Engine::threshold_e_MeV);
}
