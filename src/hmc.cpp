#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <cmath>
#include <limits>
#include <algorithm>

namespace nb = nanobind;

// Assumes C-contiguous inputs throughout

using positions_array = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::shape<-1, 2>, nb::c_contig>;
using momenta_array   = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::shape<-1, 2>, nb::c_contig>;

using positions_matrix = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using momenta_matrix   = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Vec2d            = Eigen::Vector2d;

// Utilities

static inline double wrap01(double x) {
    double y = x - std::floor(x);
    if (y >= 1.0) y -= 1.0;  // numeric guard to keep in [0,1)
    return y;
}

static inline void wrap_unit_square(Eigen::Ref<positions_matrix> X) {
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        X(i,0) = wrap01(X(i,0));
        X(i,1) = wrap01(X(i,1));
    }
}

static inline Vec2d min_image(const Vec2d& d) {
    // componentwise minimal image in (-0.5, 0.5]
    return Vec2d(d.x() - std::round(d.x()), d.y() - std::round(d.y()));
}

// Subroutines

struct Hit {
    bool  found = false;
    double t    = 0.0;
    int    i    = -1, j = -1;
    Vec2d  n    = Vec2d::Zero(); // unit normal at contact (from j to i)
};

// Earliest collision in [0, tmax] using 9 periodic images
static Hit earliest_collision(
    const Eigen::Ref<const positions_matrix>& X,
    const Eigen::Ref<const momenta_matrix>&   P,
    const double r,
    const double tmax,
    const double tol_len,
    const double tol_time
) {
    Hit best;
    if (r <= 0.0 || tmax <= 0.0) return best;

    const double R  = 2.0 * r;
    const double R2 = R * R;

    // 9 lattice shifts
    const Vec2d shifts[9] = {
        Vec2d(-1, -1), Vec2d(0, -1), Vec2d(1, -1),
        Vec2d(-1,  0), Vec2d(0,  0), Vec2d(1,  0),
        Vec2d(-1,  1), Vec2d(0,  1), Vec2d(1,  1)
    };

    const Eigen::Index n = X.rows();
    for (Eigen::Index i = 0; i < n; ++i) {
        const Vec2d xi = X.row(i);
        const Vec2d pi = P.row(i);
        for (Eigen::Index j = i + 1; j < n; ++j) {
            const Vec2d xj = X.row(j);
            const Vec2d pj = P.row(j);

            const Vec2d v = pi - pj;
            const double v_norm = v.norm();
            if (v_norm * tmax <= tol_len) continue; // negligible relative motion

            // Cheap bound via minimal-image distance at start
            const Vec2d dmi = min_image(xi - xj);
            if (dmi.norm() - v_norm * tmax >= R - tol_len) continue;

            // Check all 9 images
            for (const Vec2d& k : shifts) {
                const Vec2d d0 = (xi - xj) + k;
                const double a = v_norm * v_norm;      // v·v
                const double b = 2.0 * d0.dot(v);
                const double c = d0.squaredNorm() - R2;

                double t_cand;
                if (c <= 0.0) {
                    // Already touching/overlapping: collide now only if moving inward
                    if (b < 0.0) t_cand = 0.0; else continue;
                } else {
                    const double disc = b*b - 4.0*a*c;
                    if (disc < 0.0) continue;
                    const double sq = std::sqrt(std::max(0.0, disc));
                    const double t1 = (-b - sq) / (2.0 * a); // earliest touch
                    if (t1 < -tol_time || t1 > tmax + tol_time) continue;
                    t_cand = t1;
                }

                const Vec2d dcol = d0 + t_cand * v;
                const double nrm = dcol.norm();
                if (nrm <= tol_len) continue;

                const double t_clamped = std::min(std::max(0.0, t_cand), tmax);
                if (!best.found || t_clamped + tol_time < best.t) {
                    best.found = true;
                    best.t     = t_clamped;
                    best.i     = static_cast<int>(i);
                    best.j     = static_cast<int>(j);
                    best.n     = dcol / nrm;
                }
            }
        }
    }
    return best;
}

static inline void drift_in_place(Eigen::Ref<positions_matrix> X,
                                  const Eigen::Ref<const momenta_matrix>& P,
                                  const double tau)
{
    X.noalias() += tau * P;   // in-place update of the Python array
    wrap_unit_square(X);
}

static inline void reflect_equal_mass(Eigen::Ref<momenta_matrix> P,
                                      const int i, const int j, const Vec2d& n_hat)
{
    const Vec2d u = P.row(i) - P.row(j);
    const double u_n = u.dot(n_hat);
    if (u_n < 0.0) {
        P.row(i) -= u_n * n_hat.transpose();
        P.row(j) += u_n * n_hat.transpose();
    }
}

// Main function

void specular_reflect_torus(
    positions_array positions,
    momenta_array   momenta,
    const double    dt,
    const double    r,
    const int       max_events,
    const double    tol_len,    // e.g., 1e-12 (length units)
    const double    tol_time    // e.g., 1e-14 (time units)
) {
    if (positions.shape(0) != momenta.shape(0))
        throw std::runtime_error("positions and momenta must have the same number of rows (N).");
    if (dt <= 0.0 || r < 0.0 || max_events < 0)
        throw std::runtime_error("Invalid dt, r, or max_events.");

    Eigen::Map<positions_matrix> X(
        reinterpret_cast<double*>(positions.data()),
        static_cast<Eigen::Index>(positions.shape(0)), 2);

    Eigen::Map<momenta_matrix> P(
        reinterpret_cast<double*>(momenta.data()),
        static_cast<Eigen::Index>(momenta.shape(0)), 2);

    wrap_unit_square(X);

    double t_rem = dt;
    int events = 0;
    while (t_rem > 0.0) {
        Hit hit = earliest_collision(X, P, r, t_rem, tol_len, tol_time);
        if (!hit.found) { drift_in_place(X, P, t_rem); break; }

        if (hit.t > tol_time) { drift_in_place(X, P, hit.t); t_rem -= hit.t; }
        reflect_equal_mass(P, hit.i, hit.j, hit.n);

        if (++events >= max_events) { drift_in_place(X, P, t_rem); break; }
    }
}

// Bindings

NB_MODULE(hmc_cpp, m) {
    m.def(
        "specular_reflect_torus",
        &specular_reflect_torus,
        nb::arg("positions").noconvert(),   // (N x 2) enforced by the type
        nb::arg("momenta").noconvert(),     // (N x 2) enforced by the type
        nb::arg("dt"),
        nb::arg("r"),
        nb::arg("max_events") = 64,
        nb::arg("tol_len")    = 1e-12,
        nb::arg("tol_time")   = 1e-14
    );
}