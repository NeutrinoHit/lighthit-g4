// Light sink: the light of every charged step, kept without the steps.
//
// For LightHit's streamed showers (lighthit.streamed): per source component
// and per OM the direct light (order 0) of lighthit.direct_ring on a
// provisional clock, the earliest-arrival bound of
// lighthit.event.common_time_origins and the sums of the finest emission
// cells (lighthit.emission_cells.cell_sums).  The tables and settings come
// from lighthit.streamed.StreamPlan; the numbers follow its Python reference
// lighthit.streamed.stream_elements.  Steps are collected in blocks; a full
// block is processed by worker threads while Geant4 fills the next one.  The
// direct light of a block is summed in a fixed number of chunks in a fixed
// order, so results do not depend on the number of threads.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace lhg4 {

// everything lighthit.streamed.StreamPlan.arrays holds
struct SinkPlan {
  int components = 1;
  double reference_phase_index = 1.35, max_phase = 0, light_per_m = 0, flight_ns_per_m = 0;
  std::vector<double> pos, ori, area;   // (n,3), (n,3), (n)
  double R = 0;
  std::vector<double> n, ng, mu, eg;    // wavelength tables
  double lam0 = 0, dlam = 0;
  std::vector<double> knots, acc;
  double n_max = 0, n_min = 0, spec_max = 0, mu_min = 0;
  std::vector<double> origins;          // provisional clock P (n)
  double edge0 = 0, bin_w = 0;
  int n_bins = 0;
  std::vector<double> end_x, end_w, inner_x, inner_w;
  double piece_m = 0, pair_tol = 0;
  int chunks = 16;
  std::vector<double> s_cx, s_cy, s_dev, om_z;
  std::vector<std::int64_t> s_start, s_count, om_order;
  std::array<double, 3> centre{}, axis{}, u{}, v{};
  double delay_origin = 0;
  std::array<double, 5> cell_size{};   // axial m, transverse m, angle rad, 1/beta, delay ns

  int n_om() const { return int(area.size()); }
  void check() const;
};

struct SinkStep {
  double x, y, z, dx, dy, dz, length, c0, c2, beta, t0, t1;
  int component;
};

struct SinkChunk;

constexpr int CELL_KEYS = 7;
constexpr int CELL_SUMS = 17;

// finest-level cells of one component: open addressing on the integer keys
class CellMap {
 public:
  void add(const std::int64_t* key, const double* values);
  std::size_t size() const { return keys_.size() / CELL_KEYS; }
  const std::vector<std::int64_t>& keys() const { return keys_; }
  const std::vector<double>& sums() const { return sums_; }
  void clear();

 private:
  void grow();
  std::vector<std::int64_t> table_, keys_;
  std::vector<double> sums_;
};

struct ComponentResult {
  std::vector<double> charge, bins, skipped, earliest;   // (n), (n, n_bins), (n), (n)
  std::int64_t pairs = 0, tested = 0, steps = 0, ring_steps = 0;
  double light = 0, ring_light = 0;
  CellMap cells;
};

class LightSink {
 public:
  LightSink(SinkPlan plan, int threads);
  ~LightSink();
  LightSink(const LightSink&) = delete;
  LightSink& operator=(const LightSink&) = delete;

  // a new event: everything is cleared
  void reset();
  // one recorded vertex of a charged particle (the rows of the track format);
  // consecutive vertices of one uid make a step
  void point(std::int64_t uid, int component, double x, double y, double z, double t,
             double beta, double step_length);
  // one step after the reader's threshold (lighthit.sources.G4Shower.from_hdf5)
  void add(const SinkStep& step);
  // process what is left; results are valid afterwards
  void finish();

  const SinkPlan& plan() const { return plan_; }
  const std::vector<ComponentResult>& results() const { return results_; }
  int threads() const { return threads_; }
  std::size_t block_size = std::size_t(1) << 18;

 private:
  void submit();
  void process(std::vector<SinkStep>& block);
  void wait();

  SinkPlan plan_;
  int threads_;
  std::vector<ComponentResult> results_;
  std::vector<SinkStep> filling_, working_;
  std::vector<std::unique_ptr<SinkChunk>> parts_;   // per chunk, cleared after every block
  std::thread worker_;
  bool have_last_ = false;
  std::int64_t last_uid_ = 0;
  double last_[5] = {0, 0, 0, 0, 0};   // x, y, z, t, beta
};

}  // namespace lhg4
