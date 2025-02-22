#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;
typedef Mesh::Vertex_index vertex_descriptor; 

namespace nb = nanobind;

using vertices_array = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using faces_array = nb::ndarray<nb::numpy, int, nb::ndim<2>>;

// This class builds a mesh incrementally during surface reconstruction:
// 1. Constructor adds vertices to the mesh
// 2. Assignment operator adds faces using vertex indices
struct MeshBuilder {
    Mesh& mesh;
    
    template <typename PointIterator>
    MeshBuilder(Mesh& mesh, PointIterator b, PointIterator e)
        : mesh(mesh)
    {
        for(; b != e; ++b) {
            mesh.add_vertex(*b);
        }
    }
    
    MeshBuilder& operator=(const std::array<std::size_t, 3>& f) {
        mesh.add_face(
            vertex_descriptor(static_cast<std::size_t>(f[0])),
            vertex_descriptor(static_cast<std::size_t>(f[1])),
            vertex_descriptor(static_cast<std::size_t>(f[2]))
        );
        return *this;
    }
    
    // Iterator interface required by CGAL
    MeshBuilder& operator*() { return *this; }
    MeshBuilder& operator++() { return *this; }
    MeshBuilder operator++(int) { return *this; }
};

std::tuple<vertices_array, faces_array>
mesh_to_vertices_and_faces(const Mesh& mesh) {
    size_t n_vertices = mesh.number_of_vertices();
    size_t n_faces = mesh.number_of_faces();
    
    // Allocate raw arrays
    double* vertices_data = new double[n_vertices * 3];
    int* faces_data = new int[n_faces * 3];

    // Create memory management capsules
    nb::capsule vertices_owner(vertices_data, [](void *p) noexcept {
        delete[] (double *) p;
    });
    nb::capsule faces_owner(faces_data, [](void *p) noexcept {
        delete[] (int *) p;
    });

    // Fill vertices directly using vertex indices
    for (const auto& v : mesh.vertices()) {
        const Point& p = mesh.point(v);
        size_t idx = v.idx();  // Use built-in index
        vertices_data[idx * 3] = p.x();
        vertices_data[idx * 3 + 1] = p.y();
        vertices_data[idx * 3 + 2] = p.z();
    }

    // Fill faces using vertex indices
    for (const auto& f : mesh.faces()) {
        size_t face_idx = f.idx();
        int i = 0;
        for (vertex_descriptor v : vertices_around_face(mesh.halfedge(f), mesh)) {
            faces_data[face_idx * 3 + i] = static_cast<int>(v.idx());
            i++;
        }
    }

    // Create nanobind arrays from the raw data
    auto vertices = nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        vertices_data,
        { n_vertices, 3 },
        vertices_owner
    );
    auto faces = nb::ndarray<nb::numpy, int, nb::ndim<2>>(
        faces_data,
        { n_faces, 3 },
        faces_owner
    );

    return std::make_tuple(vertices, faces);
}

std::tuple<vertices_array, faces_array>
advancing_front_surface_reconstruction(const nb::ndarray<double>& points_array) 
{
    // Validate input
    if (points_array.ndim() != 2) {
        throw std::runtime_error("Input points must be a 2D array");
    }
    
    size_t n_points = points_array.shape(0);
    size_t dim = points_array.shape(1);
    
    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Input points must have 2 or 3 coordinates per point");
    }

    // Create vector of CGAL points efficiently
    std::vector<Point> cgal_points;
    cgal_points.reserve(n_points);
    
    const double* data = points_array.data();
    if (dim == 3) {
        for (size_t i = 0; i < n_points; i++) {
            cgal_points.emplace_back(
                data[i * 3],
                data[i * 3 + 1],
                data[i * 3 + 2]
            );
        }
    } else { // dim == 2
        for (size_t i = 0; i < n_points; i++) {
            cgal_points.emplace_back(
                data[i * 2],
                data[i * 2 + 1],
                0.0
            );
        }
    }

    Mesh mesh;
    MeshBuilder builder(mesh, cgal_points.begin(), cgal_points.end());
    
    CGAL::advancing_front_surface_reconstruction(
        cgal_points.begin(),
        cgal_points.end(),
        builder);

    return mesh_to_vertices_and_faces(mesh);
}

// Binding code
NB_MODULE(triangulation, m) {
    m.def("advancing_front_surface_reconstruction", 
          &advancing_front_surface_reconstruction,
          "Performs advancing front surface reconstruction on a point cloud\n"
          "Input: nx2 or nx3 array of points\n"
          "Returns: (vertices, faces) tuple",
          nb::arg("points"));
}