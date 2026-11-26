// Light sink (light_sink.hh).  The direct ring follows
// lighthit._direct_ring_numba (_candidates, _steps) line by line, the cell
// keys lighthit.emission_cells._keys, the earliest arrival
// lighthit._shower_basis.earliest_element_arrival.
#include "light_sink.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace lhg4 {

namespace {

constexpr double C_M_PER_NS = 0.299792458;
constexpr double PI = 3.141592653589793;

double lambda_of_n(const std::vector<double>& n_t, double lam0, double dlam, double value)
{
  if (value >= n_t[0]) return lam0;
  const std::int64_t m = std::int64_t(n_t.size()) - 1;
  if (value <= n_t[m]) return lam0 + double(m) * dlam;
  std::int64_t lo = 0, hi = m;
  while (hi - lo > 1) {
    const std::int64_t mid = (lo + hi) >> 1;
    if (n_t[mid] > value) lo = mid;
    else hi = mid;
  }
  return lam0 + (double(lo) + (n_t[lo] - value) / (n_t[lo] - n_t[hi])) * dlam;
}

int add_range(const SinkPlan& p, std::int64_t i0, std::int64_t m, double z0, double z1,
              std::int64_t* cand, int nc)
{
  std::int64_t lo = i0, hi = i0 + m;
  while (lo < hi) {
    const std::int64_t mid = (lo + hi) >> 1;
    if (p.om_z[mid] < z0) lo = mid + 1;
    else hi = mid;
  }
  for (std::int64_t t = lo; t < i0 + m && p.om_z[t] <= z1; ++t) cand[nc++] = p.om_order[t];
  return nc;
}

// OMs of the strings crossed by the cone band [th_min, th_max] (widened per string)
int candidates(const SinkPlan& p, double sx, double sy, double sz, double dx, double dy, double dz,
               double th_min, double th_max, std::int64_t* cand)
{
  const double R = p.R;
  int nc = 0;
  double pts[6];
  for (std::size_t s = 0; s < p.s_cx.size(); ++s) {
    const std::int64_t i0 = p.s_start[s], m = p.s_count[s];
    const double a = p.s_cx[s] - sx, b = p.s_cy[s] - sy;
    const double h = std::sqrt(a * a + b * b);
    const double dev = p.s_dev[s];
    if (h <= dev + R + 1.0) {
      for (std::int64_t t = 0; t < m; ++t) cand[nc++] = p.om_order[i0 + t];
      continue;
    }
    const double delta = std::asin(std::min(1.0, (R + dev) / (h - dev)));
    const double c_hi = std::cos(std::max(th_min - delta, 0.0));
    const double c_lo = std::cos(std::min(th_max + delta, PI));
    const double k = a * dx + b * dy;
    const double zlo = p.om_z[i0] - sz - 1.0;
    const double zhi = p.om_z[i0 + m - 1] - sz + 1.0;
    int npt = 2;
    pts[0] = zlo;
    pts[1] = zhi;
    for (int ci = 0; ci < 2; ++ci) {
      const double c = ci == 0 ? c_lo : c_hi;
      const double A = dz * dz - c * c;
      const double B = 2.0 * k * dz;
      const double C = k * k - c * c * h * h;
      if (std::abs(A) < 1e-14) {
        if (B != 0.0) {
          const double z = -C / B;
          if (zlo < z && z < zhi) pts[npt++] = z;
        }
      } else {
        const double disc = B * B - 4.0 * A * C;
        if (disc >= 0.0) {
          const double sq = std::sqrt(disc);
          double z = (-B - sq) / (2.0 * A);
          if (zlo < z && z < zhi) pts[npt++] = z;
          z = (-B + sq) / (2.0 * A);
          if (zlo < z && z < zhi) pts[npt++] = z;
        }
      }
    }
    for (int x = 1; x < npt; ++x) {
      const double value = pts[x];
      int y = x - 1;
      while (y >= 0 && pts[y] > value) {
        pts[y + 1] = pts[y];
        --y;
      }
      pts[y + 1] = value;
    }
    double u_keep = 0.0, v_keep = 0.0;
    bool have = false;
    for (int q = 0; q < npt - 1; ++q) {
      const double u = pts[q], v = pts[q + 1];
      if (v <= u) continue;
      const double mid = 0.5 * (u + v);
      const double f = (k + mid * dz) / std::sqrt(h * h + mid * mid);
      if (c_lo <= f && f <= c_hi) {
        if (have && u <= v_keep) {
          v_keep = v;
        } else {
          if (have) nc = add_range(p, i0, m, sz + u_keep - 1.0, sz + v_keep + 1.0, cand, nc);
          u_keep = u;
          v_keep = v;
          have = true;
        }
      }
    }
    if (have) nc = add_range(p, i0, m, sz + u_keep - 1.0, sz + v_keep + 1.0, cand, nc);
  }
  return nc;
}

}  // namespace

// accumulators of one chunk: only the touched (component, OM) rows are reduced and cleared
struct SinkChunk {
  std::vector<double> q, bins, skipped;
  std::vector<char> touched;
  std::vector<std::int64_t> rows;
  std::vector<std::int64_t> pairs, tested;
  std::vector<double> early;
  std::vector<std::int64_t> cand;

  void size(int components, int n_om, int n_bins)
  {
    q.assign(std::size_t(components) * n_om, 0.0);
    skipped.assign(q.size(), 0.0);
    bins.assign(q.size() * n_bins, 0.0);
    touched.assign(q.size(), 0);
    rows.clear();
    pairs.assign(components, 0);
    tested.assign(components, 0);
    early.assign(q.size(), std::numeric_limits<double>::infinity());
    cand.assign(4 * std::size_t(n_om), 0);   // a string may list an OM twice
  }
  void touch(std::int64_t row)
  {
    if (!touched[row]) {
      touched[row] = 1;
      rows.push_back(row);
    }
  }
  void clear(int n_bins)
  {
    for (std::int64_t row : rows) {
      q[row] = skipped[row] = 0.0;
      touched[row] = 0;
      std::fill(bins.begin() + row * n_bins, bins.begin() + (row + 1) * n_bins, 0.0);
    }
    rows.clear();
    std::fill(pairs.begin(), pairs.end(), 0);
    std::fill(tested.begin(), tested.end(), 0);
    std::fill(early.begin(), early.end(), std::numeric_limits<double>::infinity());
  }
};

namespace {

// direct ring of one step into a chunk (lighthit._direct_ring_numba._steps)
void ring_step(const SinkPlan& p, const SinkStep& e, SinkChunk& ch)
{
  const int n_om = p.n_om();
  const double b = e.beta;
  if (b * p.n_max <= 1.0) return;
  const double th_max = std::acos(1.0 / (b * p.n_max));
  const double th_min = std::acos(std::min(1.0 / (b * p.n_min), 1.0));
  const int pieces = std::max(1, int(std::ceil(e.length / p.piece_m)));
  const double dl = e.length / pieces;
  const double dt = (e.t1 - e.t0) / pieces;
  const double w0 = e.c0 / pieces, w2 = e.c2 / pieces;
  const double dx = e.dx, dy = e.dy, dz = e.dz;
  const double R = p.R;
  const std::int64_t last = std::int64_t(p.n.size()) - 2;
  const double acc_step = 2.0 / double(p.acc.size() - 1);
  const std::int64_t acc_last = std::int64_t(p.acc.size()) - 2;
  const int k_comp = e.component;
  double* q = ch.q.data() + std::size_t(k_comp) * n_om;
  double* skipped = ch.skipped.data() + std::size_t(k_comp) * n_om;
  double* bins = ch.bins.data() + std::size_t(k_comp) * n_om * p.n_bins;
  const std::size_t n_end = p.end_x.size(), n_inner = p.inner_x.size();
  for (int piece = 0; piece < pieces; ++piece) {
    const double sx = e.x + (piece + 0.5) * dl * dx;
    const double sy = e.y + (piece + 0.5) * dl * dy;
    const double sz = e.z + (piece + 0.5) * dl * dz;
    const double te = e.t0 + (piece + 0.5) * dt;
    const int nc = candidates(p, sx, sy, sz, dx, dy, dz, th_min, th_max, ch.cand.data());
    ch.tested[k_comp] += nc;
    for (int jj = 0; jj < nc; ++jj) {
      const std::int64_t j = ch.cand[jj];
      const double wx = p.pos[3 * j] - sx, wy = p.pos[3 * j + 1] - sy, wz = p.pos[3 * j + 2] - sz;
      const double r = std::sqrt(wx * wx + wy * wy + wz * wz);
      if (r <= R * 1.0001) continue;
      const double cos_a = std::min(std::max((wx * dx + wy * dy + wz * dz) / r, -1.0), 1.0);
      const double alpha = std::acos(cos_a);
      const double rho = std::asin(R / r);
      if (alpha + rho < th_min || alpha - rho > th_max) continue;
      const double a_hi = std::min(alpha + rho, 0.5 * PI - 1e-9);
      const double la = lambda_of_n(p.n, p.lam0, p.dlam, 1.0 / (b * std::cos(a_hi)));
      const double lb = lambda_of_n(p.n, p.lam0, p.dlam, 1.0 / (b * std::cos(std::max(alpha - rho, 0.0))));
      if (lb <= la) continue;
      const double c_om = -(p.ori[3 * j] * wx + p.ori[3 * j + 1] * wy + p.ori[3 * j + 2] * wz) / r;
      const double fa = (c_om + 1.0) / acc_step;
      const std::int64_t ia = std::min(std::max(std::int64_t(fa), std::int64_t(0)), acc_last);
      const double accv = p.acc[ia] + (fa - double(ia)) * (p.acc[ia + 1] - p.acc[ia]);
      const double ratio = R / r;
      const double omega = 2.0 * PI * (1.0 - std::sqrt(1.0 - ratio * ratio));
      const double pref = p.area[j] * accv * omega / (PI * R * R) / (PI * PI * rho * rho);
      if (p.pair_tol > 0.0) {
        const double bound = (pref * rho * w0 * p.spec_max * std::exp(-p.mu_min * r)
                              / std::sin(std::max(th_min, 1e-3)) * (lb - la));
        if (bound < p.pair_tol) {
          ch.touch(std::int64_t(k_comp) * n_om + j);
          skipped[j] += bound;
          continue;
        }
      }
      ch.pairs[k_comp] += 1;
      ch.touch(std::int64_t(k_comp) * n_om + j);
      double* row = bins + std::size_t(j) * p.n_bins;
      double total = 0.0;
      double p0 = la;
      std::size_t kk = 0;
      while (kk < p.knots.size() && p.knots[kk] <= la) ++kk;
      while (p0 < lb) {
        double p1 = lb;
        if (kk < p.knots.size() && p.knots[kk] < lb) {
          p1 = p.knots[kk];
          ++kk;
        }
        const double span = p1 - p0;
        const bool end_piece = (p0 == la) || (p1 == lb);
        const std::size_t nk = end_piece ? n_end : n_inner;
        for (std::size_t k = 0; k < nk; ++k) {
          double lam, jac;
          if (end_piece) {
            const double s = 0.5 * (p.end_x[k] + 1.0);
            lam = p0 + span * 0.5 * (1.0 - std::cos(PI * s));
            jac = span * 0.25 * PI * std::sin(PI * s) * p.end_w[k];
          } else {
            lam = p0 + span * 0.5 * (p.inner_x[k] + 1.0);
            jac = span * 0.5 * p.inner_w[k];
          }
          const double fi = (lam - p.lam0) / p.dlam;
          const std::int64_t ii = std::min(std::int64_t(fi), last);
          const double tt = fi - double(ii);
          const double n = p.n[ii] + tt * (p.n[ii + 1] - p.n[ii]);
          const double cth = 1.0 / (b * n);
          if (cth >= 1.0) continue;
          const double devi = std::acos(cth) - alpha;
          const double semi2 = rho * rho - devi * devi;
          if (semi2 <= 0.0) continue;
          const double mu = p.mu[ii] + tt * (p.mu[ii + 1] - p.mu[ii]);
          const double eg = p.eg[ii] + tt * (p.eg[ii + 1] - p.eg[ii]);
          const double val = ((w0 + w2 / (n * n)) / (lam * lam) * eg * std::exp(-mu * r)
                              * std::sqrt(semi2 / (1.0 - cth * cth))) * jac * pref;
          total += val;
          const double vg = C_M_PER_NS / (p.ng[ii] + tt * (p.ng[ii + 1] - p.ng[ii]));
          const double t_c = te + r / vg - p.origins[j] - p.edge0;
          const std::int64_t ib = std::int64_t(std::floor(t_c / p.bin_w));
          double early = 0.0;
          if (t_c - R / vg < double(ib) * p.bin_w) {
            const double s_e = (t_c - double(ib) * p.bin_w) * vg / R;
            early = 1.0 - s_e * s_e;
          }
          if (0 <= ib && ib < p.n_bins) row[ib] += val * (1.0 - early);
          if (0 <= ib - 1 && ib - 1 < p.n_bins) row[ib - 1] += val * early;
        }
        p0 = p1;
      }
      q[j] += total;
    }
  }
}

// lighthit._shower_basis.earliest_element_arrival for one step
void earliest_step(const SinkPlan& p, const SinkStep& e, double* earliest)
{
  const int n_om = p.n_om();
  for (int j = 0; j < n_om; ++j) {
    double dx = p.pos[3 * j] - e.x, dy = p.pos[3 * j + 1] - e.y, dz = p.pos[3 * j + 2] - e.z;
    double along = dx * e.dx + dy * e.dy + dz * e.dz;
    along = std::min(std::max(along, 0.0), e.length);
    dx -= along * e.dx;
    dy -= along * e.dy;
    dz -= along * e.dz;
    const double arrival = e.t0 + p.flight_ns_per_m * std::sqrt(dx * dx + dy * dy + dz * dz);
    if (arrival < earliest[j]) earliest[j] = arrival;
  }
}

// lighthit.emission_cells._keys of one step (finest level) and its sums
void cell_of(const SinkPlan& p, const SinkStep& e, std::int64_t* key, double* values)
{
  const double mx = e.x + 0.5 * e.length * e.dx;
  const double my = e.y + 0.5 * e.length * e.dy;
  const double mz = e.z + 0.5 * e.length * e.dz;
  const double rx = mx - p.centre[0], ry = my - p.centre[1], rz = mz - p.centre[2];
  const double z = rx * p.axis[0] + ry * p.axis[1] + rz * p.axis[2];
  const double u = rx * p.u[0] + ry * p.u[1] + rz * p.u[2];
  const double v = rx * p.v[0] + ry * p.v[1] + rz * p.v[2];
  const double along = e.dx * p.axis[0] + e.dy * p.axis[1] + e.dz * p.axis[2];
  const double theta = std::acos(std::min(std::max(along, -1.0), 1.0));
  const double phi = std::atan2(e.dx * p.v[0] + e.dy * p.v[1] + e.dz * p.v[2],
                                e.dx * p.u[0] + e.dy * p.u[1] + e.dz * p.u[2]);
  const double inverse_beta = 1.0 / e.beta;
  const double delay = 0.5 * (e.t0 + e.t1) - z / C_M_PER_NS - p.delay_origin;
  const double step = p.cell_size[2];
  const std::int64_t polar = std::int64_t(std::floor(theta / step));
  const std::int64_t rings = std::max<std::int64_t>(
    1, std::int64_t(std::nearbyint(2 * PI * std::sin((double(polar) + 0.5) * step) / step)));
  std::int64_t azimuth = std::int64_t(std::floor((phi + PI) / (2 * PI) * double(rings))) % rings;
  if (azimuth < 0) azimuth += rings;
  const double tr = p.cell_size[1];
  key[0] = std::int64_t(std::floor(z / p.cell_size[0]));
  key[1] = std::int64_t(std::floor(u / tr + 0.5));
  key[2] = std::int64_t(std::floor(v / tr + 0.5));
  key[3] = polar;
  key[4] = azimuth;
  key[5] = std::int64_t(std::floor(inverse_beta / p.cell_size[3]));
  key[6] = std::int64_t(std::floor(delay / p.cell_size[4]));
  const double w = e.c0;
  values[0] = w;
  values[1] = e.c2;
  values[2] = w * mx;
  values[3] = w * my;
  values[4] = w * mz;
  values[5] = w * e.dx;
  values[6] = w * e.dy;
  values[7] = w * e.dz;
  values[8] = w * 0.5 * (e.t0 + e.t1);
  values[9] = w * e.length;
  values[10] = w * (e.t1 - e.t0);
  values[11] = w * e.dx * e.dx;
  values[12] = w * e.dy * e.dy;
  values[13] = w * e.dz * e.dz;
  values[14] = w * e.dx * e.dy;
  values[15] = w * e.dx * e.dz;
  values[16] = w * e.dy * e.dz;
}

std::uint64_t hash_key(const std::int64_t* key)
{
  std::uint64_t code = 0x9E3779B97F4A7C15ull;
  for (int c = 0; c < CELL_KEYS; ++c) {
    code = (code ^ std::uint64_t(key[c])) * 0xBF58476D1CE4E5B9ull;
    code ^= code >> 31;
  }
  return code;
}

}  // namespace

void SinkPlan::check() const
{
  const std::size_t om = area.size();
  auto fail = [](const std::string& what) { throw std::invalid_argument("light sink plan: " + what); };
  if (components < 1) fail("components < 1");
  if (om == 0 || pos.size() != 3 * om || ori.size() != 3 * om || origins.size() != om) fail("OM arrays");
  if (n.size() < 2 || ng.size() != n.size() || mu.size() != n.size() || eg.size() != n.size())
    fail("wavelength tables");
  if (acc.size() < 3 || end_x.size() != end_w.size() || inner_x.size() != inner_w.size()
      || end_x.empty() || inner_x.empty())
    fail("acceptance or quadrature");
  if (!(bin_w > 0) || n_bins < 1 || !(piece_m > 0) || chunks < 1 || !(R > 0) || !(dlam > 0))
    fail("bins, pieces, chunks, radius or wavelength step");
  if (s_cy.size() != s_cx.size() || s_dev.size() != s_cx.size() || s_start.size() != s_cx.size()
      || s_count.size() != s_cx.size() || om_order.size() != om || om_z.size() != om)
    fail("string tables");
  for (double size : cell_size)
    if (!(size > 0)) fail("cell sizes");
  if (!(light_per_m > 0) || !(flight_ns_per_m > 0) || !(max_phase > 0) || !(reference_phase_index > 0))
    fail("light, flight or phase indices");
}

void CellMap::clear()
{
  table_.clear();
  keys_.clear();
  sums_.clear();
}

void CellMap::grow()
{
  std::size_t size = std::max<std::size_t>(1024, 2 * table_.size());
  table_.assign(size, -1);
  const std::size_t count = this->size();
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t slot = std::size_t(hash_key(&keys_[CELL_KEYS * i]) & (size - 1));
    while (table_[slot] >= 0) slot = (slot + 1) & (size - 1);
    table_[slot] = std::int64_t(i);
  }
}

void CellMap::add(const std::int64_t* key, const double* values)
{
  if (2 * (size() + 1) + 2 > table_.size()) grow();
  const std::size_t mask = table_.size() - 1;
  std::size_t slot = std::size_t(hash_key(key) & mask);
  while (true) {
    const std::int64_t j = table_[slot];
    if (j < 0) {
      table_[slot] = std::int64_t(size());
      keys_.insert(keys_.end(), key, key + CELL_KEYS);
      sums_.insert(sums_.end(), values, values + CELL_SUMS);
      return;
    }
    if (std::equal(key, key + CELL_KEYS, &keys_[CELL_KEYS * std::size_t(j)])) {
      double* row = &sums_[CELL_SUMS * std::size_t(j)];
      for (int c = 0; c < CELL_SUMS; ++c) row[c] += values[c];
      return;
    }
    slot = (slot + 1) & mask;
  }
}

LightSink::LightSink(SinkPlan plan, int threads) : plan_(std::move(plan))
{
  plan_.check();
  threads_ = threads > 0 ? threads : std::max(1u, std::thread::hardware_concurrency());
  reset();
}

LightSink::~LightSink() { wait(); }

void LightSink::wait()
{
  if (worker_.joinable()) worker_.join();
}

void LightSink::reset()
{
  wait();
  const int n_om = plan_.n_om();
  results_.assign(plan_.components, ComponentResult{});
  for (ComponentResult& r : results_) {
    r.charge.assign(n_om, 0.0);
    r.skipped.assign(n_om, 0.0);
    r.earliest.assign(n_om, std::numeric_limits<double>::infinity());
    r.bins.assign(std::size_t(n_om) * plan_.n_bins, 0.0);
  }
  filling_.clear();
  working_.clear();
  parts_.clear();
  have_last_ = false;
}

void LightSink::point(std::int64_t uid, int component, double x, double y, double z, double t,
                      double beta, double step_length)
{
  if (have_last_ && uid == last_uid_) {
    const double cx = x - last_[0], cy = y - last_[1], cz = z - last_[2];
    const double length = std::sqrt(cx * cx + cy * cy + cz * cz);
    const double b = 0.5 * (last_[4] + beta);
    if (length > 0 && step_length > 0 && b * plan_.reference_phase_index > 1) {
      const double c0 = plan_.light_per_m * step_length;
      add(SinkStep{last_[0], last_[1], last_[2], cx / length, cy / length, cz / length, length,
                   c0, -c0 / (b * b), b, last_[3], t, component});
    }
  }
  have_last_ = true;
  last_uid_ = uid;
  last_[0] = x;
  last_[1] = y;
  last_[2] = z;
  last_[3] = t;
  last_[4] = beta;
}

void LightSink::add(const SinkStep& step)
{
  if (step.component < 0 || step.component >= plan_.components)
    throw std::out_of_range("light sink: step component outside the plan");
  ComponentResult& r = results_[step.component];
  r.steps += 1;
  r.light += step.c0;
  if (step.beta * plan_.max_phase > 1) {
    r.ring_steps += 1;
    r.ring_light += step.c0;
  }
  filling_.push_back(step);
  if (filling_.size() >= block_size) submit();
}

void LightSink::submit()
{
  wait();
  std::swap(filling_, working_);
  filling_.clear();
  worker_ = std::thread([this] { process(working_); });
}

void LightSink::finish()
{
  wait();
  if (!filling_.empty()) {
    std::swap(filling_, working_);
    filling_.clear();
    process(working_);
  }
  have_last_ = false;
}

void LightSink::process(std::vector<SinkStep>& block)
{
  const SinkPlan& p = plan_;
  const int K = p.components, n_om = p.n_om();
  const std::size_t count = block.size();
  if (count == 0) return;
  // cells: one thread, in step order
  std::thread cells([&] {
    std::int64_t key[CELL_KEYS];
    double values[CELL_SUMS];
    for (const SinkStep& e : block) {
      if (!(e.beta * p.max_phase > 1)) continue;
      cell_of(p, e, key, values);
      results_[e.component].cells.add(key, values);
    }
  });
  const int chunks = int(std::min<std::size_t>(std::size_t(p.chunks), count));
  while (int(parts_.size()) < chunks) {
    parts_.push_back(std::make_unique<SinkChunk>());
    parts_.back()->size(K, n_om, p.n_bins);
  }
  std::atomic<int> next{0};
  auto work = [&] {
    for (int c = next++; c < chunks; c = next++) {
      SinkChunk& ch = *parts_[c];
      const std::size_t lo = std::size_t(c) * count / chunks, hi = std::size_t(c + 1) * count / chunks;
      for (std::size_t i = lo; i < hi; ++i) {
        const SinkStep& e = block[i];
        earliest_step(p, e, ch.early.data() + std::size_t(e.component) * n_om);
        if (e.beta * p.max_phase > 1) ring_step(p, e, ch);
      }
    }
  };
  const int helpers = std::max(0, std::min(threads_ - 1, chunks) - 1);
  std::vector<std::thread> pool;
  for (int t = 0; t < helpers; ++t) pool.emplace_back(work);
  work();
  for (std::thread& t : pool) t.join();
  // reduce in chunk order: the sums do not depend on the threads
  for (int c = 0; c < chunks; ++c) {
    SinkChunk& ch = *parts_[c];
    for (std::int64_t row : ch.rows) {
      const int k = int(row / n_om), j = int(row % n_om);
      ComponentResult& r = results_[k];
      r.charge[j] += ch.q[row];
      r.skipped[j] += ch.skipped[row];
      const double* from = ch.bins.data() + std::size_t(row) * p.n_bins;
      double* to = r.bins.data() + std::size_t(j) * p.n_bins;
      for (int b = 0; b < p.n_bins; ++b) to[b] += from[b];
    }
    for (int k = 0; k < K; ++k) {
      results_[k].pairs += ch.pairs[k];
      results_[k].tested += ch.tested[k];
      for (int j = 0; j < n_om; ++j)
        results_[k].earliest[j] = std::min(results_[k].earliest[j], ch.early[std::size_t(k) * n_om + j]);
    }
    ch.clear(p.n_bins);
  }
  cells.join();
}

}  // namespace lhg4
