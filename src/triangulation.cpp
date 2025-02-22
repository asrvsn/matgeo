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

std::tuple<nb::ndarray<double>, nb::ndarray<int>>
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
                data[i * 3],     // x
                data[i * 3 + 1], // y
                data[i * 3 + 2]  // z
            );
        }
    } else { // dim == 2
        for (size_t i = 0; i < n_points; i++) {
            cgal_points.emplace_back(
                data[i * 2],     // x
                data[i * 2 + 1], // y
                0.0             // z = 0 for 2D input
            );
        }
    }

    // Perform reconstruction
    Mesh mesh;
    CGAL::advancing_front_surface_reconstruction(
        cgal_points.begin(),
        cgal_points.end(),
        mesh);

    // Create output arrays
    size_t n_vertices = mesh.number_of_vertices();
    size_t n_faces = mesh.number_of_faces();
    
    // Create output arrays with the correct shape and layout
    std::vector<size_t> vertices_shape = {n_vertices, 3};
    std::vector<size_t> faces_shape = {n_faces, 3};
    
    auto vertices = nb::ndarray<double>(vertices_shape);
    auto faces = nb::ndarray<int>(faces_shape);

    double* vertices_data = vertices.data();
    int* faces_data = faces.data();

    // Fill vertices
    auto vertex_index = mesh.property_map<vertex_descriptor, std::size_t>("v:index").first;
    for (const auto& v : mesh.vertices()) {
        const Point& p = mesh.point(v);
        size_t idx = vertex_index[v];
        vertices_data[idx * 3] = p.x();
        vertices_data[idx * 3 + 1] = p.y();
        vertices_data[idx * 3 + 2] = p.z();
    }

    // Fill faces
    size_t face_idx = 0;
    for (const auto& f : mesh.faces()) {
        CGAL::Vertex_around_face_circulator<Mesh> vcirc(mesh.halfedge(f), mesh);
        size_t vertex_idx = 0;
        do {
            faces_data[face_idx * 3 + vertex_idx] = static_cast<int>(vertex_index[*vcirc]);
            vertex_idx++;
        } while (++vcirc != mesh.halfedge(f) && vertex_idx < 3);
        face_idx++;
    }

    return std::make_tuple(vertices, faces);
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