// Decays of chosen particles by a Python function (for example Pythia 8).
//
// Follows the Geant4 example extended/eventgenerator/pythia/py8decayer: the
// Geant4 decay tables of the chosen particles are removed, so that G4Decay
// hands their decays to the external decayer.  G4Decay uses an external
// decayer only for particles without a decay table and does not boost its
// products, so they are returned in the laboratory frame.
//
// The function is called once per decay as
//     decay(pdgid, px, py, pz, e, mass, helicity, seed)
// (GeV, laboratory frame; helicity = track polarization along the momentum;
// seed drawn from the Geant4 engine, so an event stays fixed by its seed) and
// returns a sequence of (pdgid, px, py, pz) in GeV.
#pragma once

#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "G4VExtDecayer.hh"
#include "G4VPhysicsConstructor.hh"

namespace lhg4 {

class ExternalDecayer : public G4VExtDecayer {
 public:
  explicit ExternalDecayer(nanobind::object function);
  ~ExternalDecayer() override;
  G4DecayProducts* ImportDecayProducts(const G4Track& track) override;

  long decays = 0;
  long unknown_products = 0;   // products without a Geant4 definition (dropped)
  std::string error;           // a Python exception: the event is aborted

 private:
  nanobind::object function_;
};

class ExternalDecayPhysics : public G4VPhysicsConstructor {
 public:
  ExternalDecayPhysics(nanobind::object function, std::vector<int> pdgids);
  ~ExternalDecayPhysics() override;
  void ConstructParticle() override {}
  void ConstructProcess() override;
  ExternalDecayer* decayer = nullptr;   // owned by G4Decay

 private:
  nanobind::object function_;
  std::vector<int> pdgids_;
};

}  // namespace lhg4
