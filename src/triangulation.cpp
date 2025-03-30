#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/compute_average_spacing.h>
#include "MMSurfaceNet.h"
#include "MMGeometryGL.h"
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;
typedef Mesh::Vertex_index vertex_descriptor; 

namespace nb = nanobind;

using vertices_array = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using faces_array = nb::ndarray<nb::numpy, unsigned int, nb::ndim<2>>;

// Mesh builder class for incremental surface reconstruction
// Handles vertex addition and face creation using vertex indices
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

// Converts numpy ndarray of 2D/3D points to CGAL Point_3 vector
// Handles both nx2 and nx3 arrays, setting z=0 for 2D points
std::vector<Point> convert_points_to_cgal(const nb::ndarray<double>& points_array) {
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
    
    return cgal_points;
}

// Convert CGAL mesh to numpy array of vertices
vertices_array mesh_to_vertices(const Mesh& mesh) {
    size_t n_vertices = mesh.number_of_vertices();
    
    // Allocate raw array
    double* vertices_data = new double[n_vertices * 3];

    // Create memory management capsule
    nb::capsule vertices_owner(vertices_data, [](void *p) noexcept {
        delete[] (double *) p;
    });

    // Fill vertices directly using vertex indices
    for (const auto& v : mesh.vertices()) {
        const Point& p = mesh.point(v);
        size_t idx = v.idx();  // Use built-in index
        vertices_data[idx * 3] = p.x();
        vertices_data[idx * 3 + 1] = p.y();
        vertices_data[idx * 3 + 2] = p.z();
    }

    return vertices_array(vertices_data, { n_vertices, 3 }, vertices_owner);
}

// Convert CGAL mesh to numpy array of faces
faces_array mesh_to_faces(const Mesh& mesh) {
    size_t n_faces = mesh.number_of_faces();
    
    // Allocate raw array with the correct type
    unsigned int* faces_data = new unsigned int[n_faces * 3];

    // Create memory management capsule
    nb::capsule faces_owner(faces_data, [](void *p) noexcept {
        delete[] (unsigned int *) p;
    });

    // Fill faces using vertex indices
    for (const auto& f : mesh.faces()) {
        size_t face_idx = f.idx();
        int i = 0;
        for (vertex_descriptor v : vertices_around_face(mesh.halfedge(f), mesh)) {
            faces_data[face_idx * 3 + i] = static_cast<unsigned int>(v.idx());
            i++;
        }
    }

    return faces_array(faces_data, { n_faces, 3 }, faces_owner);
}

// Performs advancing front surface reconstruction on point cloud
// Takes nx2 or nx3 point array, returns faces array
faces_array advancing_front_surface_reconstruction(const nb::ndarray<double>& points_array) 
{
    std::vector<Point> cgal_points = convert_points_to_cgal(points_array);
    Mesh mesh;
    MeshBuilder builder(mesh, cgal_points.begin(), cgal_points.end());
    CGAL::advancing_front_surface_reconstruction(
        cgal_points.begin(),
        cgal_points.end(),
        builder);
    
    return mesh_to_faces(mesh);
}

// Define the point-normal pair type
typedef std::pair<Point, Kernel::Vector_3> PointVectorPair;

// Converts numpy ndarrays of points and normals to vector of CGAL point-normal pairs
std::vector<PointVectorPair> convert_points_normals_to_cgal(
    const nb::ndarray<double>& points_array,
    const nb::ndarray<double>& normals_array) 
{
    // Validate input
    if (points_array.ndim() != 2 || normals_array.ndim() != 2) {
        throw std::runtime_error("Input points and normals must be 2D arrays");
    }
    if (points_array.shape(0) != normals_array.shape(0) || 
        points_array.shape(1) != 3 || normals_array.shape(1) != 3) {
        throw std::runtime_error("Points and normals must be nx3 arrays of same length");
    }

    // Create vector of point-normal pairs
    std::vector<PointVectorPair> points;
    size_t n_points = points_array.shape(0);
    points.reserve(n_points);

    const double* points_data = points_array.data();
    const double* normals_data = normals_array.data();

    for (size_t i = 0; i < n_points; i++) {
        points.emplace_back(
            Point(
                points_data[i * 3],
                points_data[i * 3 + 1],
                points_data[i * 3 + 2]
            ),
            Kernel::Vector_3(
                normals_data[i * 3],
                normals_data[i * 3 + 1],
                normals_data[i * 3 + 2]
            )
        );
    }
    
    return points;
}

faces_array poisson_surface_reconstruction(
    const nb::ndarray<double>& points_array,
    const nb::ndarray<double>& normals_array)
{
    std::vector<PointVectorPair> points = convert_points_normals_to_cgal(points_array, normals_array);

    // Compute average spacing
    double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
        points,
        6,  // number of neighbors
        CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())
    );

    // Create mesh
    Mesh mesh;
    CGAL::poisson_surface_reconstruction_delaunay(
        points.begin(),
        points.end(),
        CGAL::First_of_pair_property_map<PointVectorPair>(),
        CGAL::Second_of_pair_property_map<PointVectorPair>(),
        mesh,
        average_spacing
    );

    return mesh_to_faces(mesh);
}

// Function to extract surface mesh from labeled volume using Surface Nets
std::tuple<vertices_array, faces_array> 
extract_surface_net(
    const nb::ndarray<nb::numpy, unsigned short, nb::ndim<3>, nb::c_contig>& labels_array, 
    const nb::ndarray<nb::numpy, float, nb::shape<3>, nb::c_contig>& voxel_size_array) 
{
    int array_size[3] = {
        static_cast<int>(labels_array.shape(0)), static_cast<int>(labels_array.shape(1)), static_cast<int>(labels_array.shape(2))
    };
    float voxel_size[3] = {
        voxel_size_array.data()[0], voxel_size_array.data()[1], voxel_size_array.data()[2]
    };
    
    // Create Surface Net with raw pointer to the numpy array data
    MMSurfaceNet surface_net(
        const_cast<unsigned short*>(labels_array.data()), // Cast away const for the API
        array_size,
        voxel_size
    );

     // Create MMGeometryGL to convert the surface net to vertex and index arrays
    MMGeometryGL geometry(&surface_net);

    // Get the vertex and index data from the geometry
    size_t num_vertices = static_cast<size_t>(geometry.numVertices());
    size_t num_indices = static_cast<size_t>(geometry.numIndices());
    float* gl_vertices = geometry.vertices();
    unsigned int* gl_indices = geometry.indices();
    
    // Create vertices array (extract positions from the GL vertices)
    double* vertices_data = new double[num_vertices * 3];
    nb::capsule vertices_owner(vertices_data, [](void *p) noexcept {
        delete[] (double *) p;
    });

    // Extract positions from GL vertices - use direct indexing for efficiency
    for (size_t i = 0, v_idx = 0, gl_idx = 0; i < num_vertices; i++, gl_idx += 8) {
        vertices_data[v_idx++] = gl_vertices[gl_idx];     // pos[0]
        vertices_data[v_idx++] = gl_vertices[gl_idx + 1]; // pos[1]
        vertices_data[v_idx++] = gl_vertices[gl_idx + 2]; // pos[2]
    }
    
    // Create faces array (indices are already in triangle format)
    size_t num_triangles = num_indices / 3;
    unsigned int* faces_data = new unsigned int[num_indices];
    nb::capsule faces_owner(faces_data, [](void *p) noexcept {
        delete[] (unsigned int *) p;
    });
    
    // Copy indices - use memcpy for efficiency since types match
    std::memcpy(faces_data, gl_indices, num_indices * sizeof(unsigned int));
    
    return std::tuple<vertices_array, faces_array>(
        vertices_array(vertices_data, { num_vertices, 3 }, vertices_owner),
        faces_array(faces_data, { num_triangles, 3 }, faces_owner)
    );
}

// // Add relaxation parameters to the surface net extraction
// std::tuple<vertices_array, faces_array> 
// extract_surface_net_with_relaxation(
//     const nb::ndarray<nb::numpy, unsigned short, nb::ndim<3>, nb::c_contig>& labels_array, 
//     const nb::ndarray<nb::numpy, float, nb::shape<3>, nb::c_contig>& voxel_size_array,
//     int num_relax_iterations = 5,
//     float relax_factor = 0.5,
//     float max_dist_from_cell_center = 0.45) 
// {
//     // Get array dimensions
//     int array_size[3] = {
//         static_cast<int>(labels_array.shape(0)),
//         static_cast<int>(labels_array.shape(1)),
//         static_cast<int>(labels_array.shape(2))
//     };
    
//     // Create Surface Net
//     MMSurfaceNet surface_net(
//         const_cast<unsigned short*>(labels_array.data()), // Cast away const for the API
//         array_size,
//         const_cast<float*>(voxel_size_array.data()) // Cast away const for the API
//     );
    
//     // Set relaxation attributes
//     MMSurfaceNet::RelaxAttrs relax_attrs;
//     relax_attrs.numRelaxIterations = num_relax_iterations;
//     relax_attrs.relaxFactor = relax_factor;
//     relax_attrs.maxDistFromCellCenter = max_dist_from_cell_center;
    
//     // Perform relaxation
//     surface_net.relax(relax_attrs);
    
//     // For now, return placeholder empty arrays
//     // We'll implement the actual mesh extraction in the next step
    
//     // Create empty vertices array
//     double* vertices_data = new double[1 * 3];
//     vertices_data[0] = vertices_data[1] = vertices_data[2] = 0.0;
//     nb::capsule vertices_owner(vertices_data, [](void *p) noexcept {
//         delete[] (double *) p;
//     });
    
//     // Create empty faces array
//     int* faces_data = new int[1 * 3];
//     faces_data[0] = faces_data[1] = faces_data[2] = 0;
//     nb::capsule faces_owner(faces_data, [](void *p) noexcept {
//         delete[] (int *) p;
//     });
    
//     return std::make_tuple(
//         vertices_array(vertices_data, { 1, 3 }, vertices_owner),
//         faces_array(faces_data, { 1, 3 }, faces_owner)
//     );
// }

// // Get unique labels from the surface net
// nb::ndarray<nb::numpy, int> 
// get_surface_net_labels(
//     const nb::ndarray<nb::numpy, unsigned short, nb::ndim<3>, nb::c_contig>& labels_array, 
//     const nb::ndarray<nb::numpy, float, nb::shape<3>, nb::c_contig>& voxel_size_array) 
// {
//     // Get array dimensions
//     int array_size[3] = {
//         static_cast<int>(labels_array.shape(0)),
//         static_cast<int>(labels_array.shape(1)),
//         static_cast<int>(labels_array.shape(2))
//     };
    
//     // Create Surface Net
//     MMSurfaceNet surface_net(
//         const_cast<unsigned short*>(labels_array.data()),
//         array_size,
//         const_cast<float*>(voxel_size_array.data()) // Cast away const for the API
//     );
    
//     // Get unique labels
//     std::vector<int> unique_labels = surface_net.labels();
    
//     // Create output array
//     size_t num_labels = unique_labels.size();
//     int* labels_data = new int[num_labels];
    
//     // Copy labels to output array
//     for (size_t i = 0; i < num_labels; i++) {
//         labels_data[i] = unique_labels[i];
//     }
    
//     // Create memory management capsule
//     nb::capsule labels_owner(labels_data, [](void *p) noexcept {
//         delete[] (int *) p;
//     });
    
//     return nb::ndarray<nb::numpy, int>(labels_data, { num_labels }, labels_owner);
// }

// Binding code
NB_MODULE(triangulation_cpp, m) {
    m.def("advancing_front_surface_reconstruction", 
          &advancing_front_surface_reconstruction,
          "Performs advancing front surface reconstruction on a point cloud\n"
          "Input: nx2 or nx3 array of points\n"
          "Returns: faces array (triangles defined by vertex indices)",
          nb::arg("points").noconvert());
          
    m.def("poisson_surface_reconstruction",
          &poisson_surface_reconstruction,
          "Performs Poisson surface reconstruction on a point cloud with normals\n"
          "Input: nx3 array of points and nx3 array of normals\n"
          "Returns: faces array (triangles defined by vertex indices)",
          nb::arg("points").noconvert(),
          nb::arg("normals").noconvert());
          
    m.def("extract_surface_net",
          &extract_surface_net,
          "Extracts a surface mesh from a labeled volume using Surface Nets\n"
          "Input: 3D array of labels (unsigned short) and 3-element array of voxel sizes\n"
          "Returns: tuple of vertices and faces arrays",
          nb::arg("labels").noconvert(),
          nb::arg("voxel_size"));
          
    // Uncomment these when you're ready to expose these functions
    /*
    m.def("extract_surface_net_with_relaxation",
          &extract_surface_net_with_relaxation,
          "Extracts a surface mesh from a labeled volume using Surface Nets with relaxation\n"
          "Input: 3D array of labels (unsigned short), 3-element array of voxel sizes, and relaxation parameters\n"
          "Returns: tuple of vertices and faces arrays",
          nb::arg("labels").noconvert(),
          nb::arg("voxel_size").noconvert(),
          nb::arg("num_relax_iterations") = 5,
          nb::arg("relax_factor") = 0.5,
          nb::arg("max_dist_from_cell_center") = 0.45);
          
    m.def("get_surface_net_labels",
          &get_surface_net_labels,
          "Gets the unique labels from a labeled volume\n"
          "Input: 3D array of labels (unsigned short) and 3-element array of voxel sizes\n"
          "Returns: array of unique labels",
          nb::arg("labels").noconvert(),
          nb::arg("voxel_size").noconvert());
    */
}