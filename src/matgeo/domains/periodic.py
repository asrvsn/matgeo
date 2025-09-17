'''
A collection of periodic domain construction helpers
'''
import numpy as np
import pygmsh
import gmsh
import meshio
from mpi4py import MPI
import dolfinx.io
import dolfinx.mesh
import dolfinx_mpc
import pdb
import basix
import basix.ufl
import ufl
from typing import Tuple

from gmshModel.Model.RandomInclusionRVE import RandomInclusionRVE

from .utils import *

class SwissCheeseRVE(RandomInclusionRVE):
    """
    Swiss cheese RVE with explicitly specified circular inclusion centers and radii.
    Subclasses gmshModel's RandomInclusionRVE to create periodic domains with holes.
    """
    
    def __init__(self, centers: np.ndarray, radii: np.ndarray, 
                 size=[1.0, 1.0, 0.0], mesh_size=0.1, **kwargs):
        """
        Initialize Swiss cheese RVE with explicit inclusion positions.
        
        Args:
            centers: (n, 2) array of inclusion centers in [0,1)^2
            radii: (n,) array of inclusion radii
            size: Domain size [width, height, depth]
            mesh_size: Characteristic mesh size
            **kwargs: Additional arguments passed to RandomInclusionRVE
        """
        super().__init__(
            size=size,
            inclusionSets=np.array([[0.01, 0.001]]),  # Small dummy inclusion set
            inclusionType="Circle",
            origin=[0.0, 0.0, 0.0],
            periodicityFlags=[1, 1, 0],  # Periodic in x and y
            domainGroup='domain',
            inclusionGroup='inclusions',
            **kwargs
        )
        
        self.centers = np.asarray(centers, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.mesh_size = mesh_size
        
        # Validate inputs
        assert self.centers.ndim == 2 and self.centers.shape[1] == 2
        assert self.radii.ndim == 1 and self.radii.shape[0] == self.centers.shape[0]
        assert (self.centers.min() >= 0.0) and (self.centers.max() < 1.0), "Centers must lie in [0,1)^2"
    
    def placeInclusions(self, placementOptions={}):
        """
        Override parent method to place inclusions at explicit centers with given radii.
        Also handles periodic images of inclusions that overlap domain boundaries.
        """
        # Build inclusion info array: [x, y, z, radius]
        inclusion_info = []
        
        # For each inclusion, create periodic images if they intersect unit square boundaries
        for c, r in zip(self.centers, self.radii):
            for dx in [-1.0, 0.0, 1.0]:
                for dy in [-1.0, 0.0, 1.0]:
                    c_img = c + np.array([dx, dy])
                    # Check if this periodic image intersects the unit square [0,1]^2
                    if self._circle_intersects_unit_square(c_img, r):
                        # Add origin offset to match RVE coordinate system
                        center_3d = c_img + np.array(self.origin[:2])
                        inclusion_info.append([center_3d[0], center_3d[1], self.origin[2], float(r)])
        
        # Set inclusionInfo (the only thing the parent class needs from us)
        if inclusion_info:
            self.inclusionInfo = np.array(inclusion_info)
        else:
            self.inclusionInfo = np.empty((0, 4))
    
    def _circle_intersects_unit_square(self, center: np.ndarray, radius: float) -> bool:
        """Check if a circle intersects the unit square [0,1]^2."""
        return (center[0] + radius > 0.0 and center[0] - radius < 1.0 and 
                center[1] + radius > 0.0 and center[1] - radius < 1.0)
    
    
    def generate_mesh(self, **mesh_options):
        """Generate the mesh with appropriate sizing."""
        # Set mesh size options
        if 'characteristicLength' not in mesh_options:
            mesh_options['characteristicLength'] = self.mesh_size
        
        return super().generate_mesh(**mesh_options)

    def to_dolfinx(self) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.meshtags, dolfinx.mesh.meshtags]:
        return dolfinx.io.gmshio.model_to_mesh(self.gmshAPI, MPI.COMM_WORLD, 0, gdim=self.dimension)

    def get_physical_mesh(self, physical_group: int):
        """
        Convert the current gmsh model to a DOLFiN-X mesh and return a submesh
        consisting only of cells in the given physical group.
        
        Returns:
            dolfinx.mesh.Mesh: Submesh of the physical group
        """
        mesh, cell_tags, _ = self.to_dolfinx()
        assert cell_tags is not None and len(cell_tags.indices) > 0, "Cell tags are not available"
        cells_pg = cell_tags.indices[cell_tags.values == physical_group]
        assert len(cells_pg) > 0, "No cells in physical group"
        tdim = mesh.topology.dim
        submesh, _, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, cells_pg)
        return submesh

    def setup_dolfinx(self, element: basix.ufl.element) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.meshtags, ufl.Measure, dolfinx.fem.functionspace, dolfinx_mpc.MultiPointConstraint]:
        '''
        Get everything you need to solve a dolfinx problem on the domain.
        '''
        # 1. Make mesh
        mesh = self.get_physical_mesh(1) # Domain
        tdim = mesh.topology.dim
        assert tdim == 2, "Only 2D meshes are supported"
        mesh.topology.create_connectivity(tdim-1, 0)
        mesh.topology.create_connectivity(tdim-1, tdim)
        e2v = mesh.topology.connectivity(tdim-1, 0)

        # 1. Split exterior facets into periodic boundary and holes
        ext_indices = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        centroids = np.array([
            mesh.geometry.x[e2v.links(f)].mean(axis=0) for f in ext_indices
        ])
        ID_HOLES, ID_LEFT, ID_RIGHT, ID_TOP, ID_BOTTOM = 1, 2, 3, 4, 5
        def label_centroid(cntr: np.ndarray) -> int:
            if np.isclose(cntr[0], 0.0): return ID_LEFT
            elif np.isclose(cntr[0], 1.0): return ID_RIGHT
            elif np.isclose(cntr[1], 1.0): return ID_TOP
            elif np.isclose(cntr[1], 0.0): return ID_BOTTOM
            else: return ID_HOLES
        labels = np.array([label_centroid(c) for c in centroids], dtype=np.int32)
        ext_facetags = dolfinx.mesh.meshtags(mesh, tdim-1, ext_indices, labels)

        # 2. Create boundary measure for holes
        ds = ufl.Measure('ds', domain=mesh, subdomain_data=ext_facetags)(1)
        
        # 2. Create function space and multi-point constraint for periodicity
        V = dolfinx.fem.functionspace(mesh, element)
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_topological( # Left-right
            V, ext_facetags, ID_LEFT, lambda x: np.array([x[0]-1, x[1]]), []
        )
        mpc.create_periodic_constraint_topological( # Top-bottom
            V, ext_facetags, ID_TOP, lambda x: np.array([x[0], x[1]-1]), []
        )
        mpc.finalize()

        return mesh, ext_facetags, ds, V, mpc

def swiss_cheese(
        centers: np.ndarray, 
        radii: np.ndarray, 
        boundary_elements: int=40,
        **kwargs
    ) -> SwissCheeseRVE:
    """
    Creates a 2D triangulation for a periodic domain [0, 1)^2 with circular holes.
    Holes are periodized: if a disk overlaps an edge, its image appears on the opposite side.
    
    Args:
        centers: (n, 2) array of inclusion centers in [0,1)^2
        radii: (n,) array of inclusion radii
        lcar: Mesh size
        **kwargs: Additional arguments passed to SwissCheeseRVE
    
    Returns:
        meshio.Mesh: Generated mesh
    """
    rve = SwissCheeseRVE(
        centers=centers,
        radii=radii,
        size=[1.0, 1.0, 0.0],
        **kwargs
    )
    rve.createGmshModel()
    meshingParameters = {
        "threads": None,
        "refinementOptions": {
            "maxMeshSize": "auto",
            "inclusionRefinement": True,
            "interInclusionRefinement": True,
            "elementsPerCircumference": boundary_elements,
            "elementsBetweenInclusions": 3,
            "inclusionRefinementWidth": 3,
            "transitionElements": "auto",
            # "aspectRatio": 2,
        }
    }
    rve.createMesh(**meshingParameters)
    return rve

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import asrvsn_mpl as ampl
    from ..points.matern import matern_II_torus
    from ..points.cvt import gradient_flow_cvt_torus

    rng = np.random.default_rng(42)

    r = 0.1
    xs = matern_II_torus(100, r, rng=rng)
    xs = gradient_flow_cvt_torus(xs, 10, 1.0)
    # mesh = swiss_cheese(xs, np.full(xs.shape[0], r), 0.01, 0.1)
    # tri = Triangulation.from_gmsh(mesh)

    # fig, ax = plt.subplots()
    # ampl.ax_tri_2d(ax, tri.pts, tri.simplices)
    # plt.show()
    rve = swiss_cheese(xs, np.full(xs.shape[0], r), boundary_elements=50)
    print(f'True boundary length: {xs.shape[0] * 2 * np.pi * r}')
    # rve.visualizeMesh()

    # domain_mesh = rve.get_physical_mesh(2)
    # visualize_exterior_tags(domain_mesh)

    # rve.setup_dolfinx(('Lagrange', 1))
    # mesh, cell_tags, facet_tags = rve.to_dolfinx()
    # sub_mesh, _, _, _ = get_submesh_by_cell(mesh, cell_tags, 1)

    mesh, ft, ds, V, mpc = rve.setup_dolfinx(('Lagrange', 1)) 
    print(f'Dolfinx boundary length: {compute_scalar(1.0 * ds)}')
    visualize_tags(mesh, ft)  