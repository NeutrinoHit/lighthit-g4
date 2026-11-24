#include "external_decayer.hh"

#include <algorithm>
#include <cstdint>

#include "G4Decay.hh"
#include "G4DecayProducts.hh"
#include "G4DecayTable.hh"
#include "G4DynamicParticle.hh"
#include "G4EventManager.hh"
#include "G4Exception.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "Randomize.hh"

namespace nb = nanobind;

namespace lhg4 {

ExternalDecayer::ExternalDecayer(nb::object function)
  : G4VExtDecayer("ExternalDecayer"), function_(std::move(function))
{}

// Geant4 may delete these objects after the interpreter has finished: the
// reference to the Python function is then left alone, never released.
ExternalDecayer::~ExternalDecayer() { function_.release(); }

G4DecayProducts* ExternalDecayer::ImportDecayProducts(const G4Track& track)
{
  const G4DynamicParticle* parent = track.GetDynamicParticle();
  const G4ParticleDefinition* definition = track.GetDefinition();
  const G4ThreeVector momentum = track.GetMomentum();
  const double helicity = std::max(
    -1.0, std::min(1.0, track.GetPolarization().dot(track.GetMomentumDirection())));
  // seeds of the Marsaglia-Zaman generator of Pythia lie in [0, 900000000]
  const long seed = static_cast<long>(G4UniformRand() * 900000000.0);
  auto* products = new G4DecayProducts(*parent);
  if (!error.empty()) return products;
  nb::gil_scoped_acquire gil;
  try {
    nb::object result = function_(definition->GetPDGEncoding(), momentum.x() / GeV,
                                  momentum.y() / GeV, momentum.z() / GeV,
                                  parent->GetTotalEnergy() / GeV, definition->GetPDGMass() / GeV,
                                  helicity, seed);
    auto* table = G4ParticleTable::GetParticleTable();
    for (nb::handle item : result) {
      const int code = nb::cast<int>(item[0]);
      G4ParticleDefinition* daughter = table->FindParticle(code);
      if (daughter == nullptr) {
        ++unknown_products;
        continue;
      }
      const G4ThreeVector p(nb::cast<double>(item[1]) * GeV, nb::cast<double>(item[2]) * GeV,
                            nb::cast<double>(item[3]) * GeV);
      products->PushProducts(new G4DynamicParticle(daughter, p));
    }
    ++decays;
  } catch (const std::exception& exc) {
    error = exc.what();
    G4EventManager::GetEventManager()->AbortCurrentEvent();
  }
  return products;
}

ExternalDecayPhysics::ExternalDecayPhysics(nb::object function, std::vector<int> pdgids)
  : G4VPhysicsConstructor("ExternalDecayPhysics"),
    function_(std::move(function)),
    pdgids_(std::move(pdgids))
{}

ExternalDecayPhysics::~ExternalDecayPhysics() { function_.release(); }

void ExternalDecayPhysics::ConstructProcess()
{
  G4Decay* decay = nullptr;
  auto* table = G4ParticleTable::GetParticleTable();
  for (int code : pdgids_) {
    G4ParticleDefinition* particle = table->FindParticle(code);
    if (particle == nullptr)
      G4Exception("ExternalDecayPhysics", "LHG4D01", FatalException, "unknown particle");
    if (particle->GetDecayTable() != nullptr) {
      delete particle->GetDecayTable();
      particle->SetDecayTable(nullptr);
    }
    G4ProcessVector* processes = particle->GetProcessManager()->GetProcessList();
    for (std::size_t i = 0; i < processes->size(); ++i)
      if (auto* candidate = dynamic_cast<G4Decay*>((*processes)[i])) decay = candidate;
  }
  if (decay == nullptr)
    G4Exception("ExternalDecayPhysics", "LHG4D02", FatalException, "no G4Decay process found");
  // one G4Decay object serves all particles; it owns and deletes the decayer
  decayer = new ExternalDecayer(function_);
  decay->SetExtDecayer(decayer);
}

}  // namespace lhg4
