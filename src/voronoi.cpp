#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "voro++.hh"

#include <iostream>
#include <cmath>

namespace nb = nanobind;

using namespace nb::literals;

// Define types 
typedef nb::ndarray<double, nb::ndim<2>> point_data;

voro::container_periodic make_2d_box(point_data xs, const double L) {
    assert(xs.shape(1) == 2);
    const int N = static_cast<int>(xs.shape(0));
    const double V = L * L; // L x L x 1

    // Choose blocks using heuristic as in freud
    // https://github.com/glotzerlab/freud/blob/main/cpp/locality/Voronoi.cc
    const double block_scale = std::pow(N / (voro::optimal_particles * V), 1.0 / 3.0);
    const int bks_xy = static_cast<int>(L * block_scale + 1); 
    const int bks_z = static_cast<int>(1 * block_scale + 1);
    voro::container_periodic box(L, 0, L, 0, 0, 1, bks_xy, bks_xy, bks_z, 3);

    auto xsv = xs.view(); 
    for (int i = 0; i < N; ++i) {
        box.put(i, xsv(i, 0), xsv(i, 1), 0.5); // Place points in plane at z=0.5
    }
    return box;
}

/**
 * A function to calculate 2D forces based on positions and length.
 *
 * @param xs Nx2 list of 2D positions
 * @param fs Nx2 list of 2D forces to be populated
 * @param L the periodic box side length
 *
 * @throws None
 */
void force_2d(point_data xs, point_data fs, const double L) {
    assert(xs.shape(0) == fs.shape(0));
    assert(xs.shape(1) == fs.shape(1));
    voro::container_periodic box = make_2d_box(xs, L);

    auto xsv = xs.view();
    auto fsv = fs.view();
    voro::voronoicell cell;
    voro::c_loop_all_periodic vloop(box);
    if (vloop.start()) do {
        box.compute_cell(cell, vloop);
        double area = cell.volume();
        double cx, cy, cz;
        cell.centroid(cx, cy, cz);
        int j = vloop.pid();

        // Voronoi liquid force
        fsv(j, 0) = 2 * area * (cx - xsv(j, 0));
        fsv(j, 1) = 2 * area * (cy - xsv(j, 1));
    } while (vloop.inc());
}

/**
 * A function to calculate 2D areas based on positions and length.
 *
 * @param xs Nx2 list of 2D positions
 * @param areas 1D array of areas to be populated
 * @param L the periodic box side length
 *
 * @throws None
 */
void areas_2d(point_data xs, nb::ndarray<double, nb::ndim<1>> areas, const double L) {
    assert(xs.shape(0) == areas.shape(0));
    voro::container_periodic box = make_2d_box(xs, L);

    auto asv = areas.view();
    voro::voronoicell cell;
    voro::c_loop_all_periodic vloop(box);
    if (vloop.start()) do {
        box.compute_cell(cell, vloop);
        asv(vloop.pid()) = cell.volume(); // Box has z length 1
    } while (vloop.inc());
}

void centroids_2d(point_data xs, point_data cs, const double L) {
    assert(xs.shape(0) == cs.shape(0));
    assert(xs.shape(1) == cs.shape(1));
    voro::container_periodic box = make_2d_box(xs, L);

    auto xsv = xs.view();
    auto csv = cs.view();
    voro::voronoicell cell;
    voro::c_loop_all_periodic vloop(box);
    if (vloop.start()) do {
        box.compute_cell(cell, vloop);
        std::vector<double> vertices;
        cell.vertices(vertices);
        double cx = 0, cy = 0;
        double vxi = 0, vyi = 0;  
        double vxj = 0, vyj = 0;  
        int j;
        int nv = static_cast<int>(vertices.size()) / 3;  // vertices has 3 components per vertex
        // assert(nv == cell.number_of_faces() - 2);
        // std::cout << "nv = " << nv << std::endl;
        // Compute centroids manually using shoelace formula
        for (int i = 0; i < nv; ++i) {
            j = (i + 1) % nv;  // Next vertex (consecutive vertices for shoelace)
            vxi = vertices[i*3];     
            vyi = vertices[i*3+1];   
            vxj = vertices[j*3];     
            vyj = vertices[j*3+1];   
            // std::cout << vxi << " " << vyi << std::endl;
            cx += (vxi + vxj) * (vxi * vyj - vxj * vyi);
            cy += (vyi + vyj) * (vxi * vyj - vxj * vyi);
        }
        double nc = 6 * cell.volume(); // Normalizing constant
        cx /= nc;
        cy /= nc;
        // std::cout << cx << " " << cy << std::endl;
        // double cx, cy, cz;
        // cell.centroid(cx, cy, cz);
        int n = vloop.pid();
        csv(n, 0) = cx;
        csv(n, 1) = cy;
    } while (vloop.inc());
}

void degree_2d(point_data xs, nb::ndarray<int, nb::ndim<1>> ds, const double L) {
    assert(xs.shape(0) == ds.shape(0));
    voro::container_periodic box = make_2d_box(xs, L);

    auto xsv = xs.view();
    auto dsv = ds.view();
    voro::voronoicell cell;
    voro::c_loop_all_periodic vloop(box);
    if (vloop.start()) do {
        box.compute_cell(cell, vloop);
        dsv(vloop.pid()) = cell.number_of_faces() - 2; // Two periodic faces in Z
    } while (vloop.inc());
}

NB_MODULE(voronoi_cpp, m) {
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("force_2d", &force_2d);
    m.def("areas_2d", &areas_2d);
    m.def("centroids_2d", &centroids_2d);
    m.def("degree_2d", &degree_2d);
}
