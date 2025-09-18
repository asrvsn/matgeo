'''
Utilities for dolfinx mesh
'''
from typing import Tuple, List
import numpy as np
import dolfinx.mesh
import dolfinx.plot
import pyvista
from ufl.core.expr import Expr as UFLExpr
import dolfinx.fem

def get_exterior_tags(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.meshtags:
    '''
    Get the tags for the inclusions in the mesh
    '''
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    vals = np.ones(len(facets), dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim-1, facets, vals)

def visualize_tags(mesh: dolfinx.mesh.Mesh, tags: dolfinx.mesh.meshtags, cmap='glasbey'):
    tdim = mesh.topology.dim
    # Interior
    topo, types, geom = dolfinx.plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topo, types, geom)
    # Exterior
    tags_topo, tags_types, tags_geom = dolfinx.plot.vtk_mesh(mesh, tags.dim)
    tags_grid = pyvista.UnstructuredGrid(tags_topo, tags_types, tags_geom)
    tag_data = np.full(tags_grid.n_cells, -1, dtype=np.int32)
    tag_data[tags.indices] = tags.values
    tags_grid.cell_data['FacetTags'] = tag_data
    # Plot
    p = pyvista.Plotter(window_size=(1000, 1000))
    p.add_mesh(grid, show_edges=False, opacity=0.2)
    p.add_mesh(tags_grid, scalars='FacetTags', line_width=2, render_lines_as_tubes=True, show_scalar_bar=True, lighting=False, cmap=cmap)
    p.view_xy()
    p.reset_camera()
    p.show()

def visualize_field(mesh: dolfinx.mesh.Mesh, field: dolfinx.fem.Function, warp: bool=True):
    '''
    Visualize a field on a mesh
    '''
    tdim = mesh.topology.dim
    topo, types, geom = dolfinx.plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topo, types, geom)
    grid.point_data['Field'] = field.x.array
    grid.set_active_scalars('Field')
    p = pyvista.Plotter(window_size=(2000, 2000))
    p.add_mesh(grid, scalars='Field', show_edges=True)
    p.view_xy()
    p.reset_camera()
    if warp:
        warped = grid.warp_by_scalar()
        p.add_mesh(warped)
    p.show()

def visualize_periodic_domain(mesh: dolfinx.mesh.Mesh, ft: dolfinx.mesh.meshtags, assoc: List[Tuple[int, int]]):
    tdim = mesh.topology.dim
    cells_topo, cell_types, geom = dxf_plot.vtk_mesh(mesh, tdim)
    grid = pv.UnstructuredGrid(cells_topo, cell_types, geom)
    f_topo, f_types, f_geom = dolfinx.plot.vtk_mesh(mesh, tdim-1)
    fgrid = pv.UnstructuredGrid(f_topo, f_types, f_geom)

    # attach facet tags to the facet grid
    tags = np.full(fgrid.n_cells, -1, dtype=np.int32)
    tags[ft.indices] = ft.values
    fgrid.cell_data["FacetTags"] = tags

    # facet centroids (for arrows)
    centroids = get_facet_centroids(mesh, np.arange(fgrid.n_cells, dtype=np.int32))
    fgrid.point_data["Centroids"] = centroids

    # bbox and cell lengths
    bbmin = mesh.geometry.x.min(axis=0)
    bbmax = mesh.geometry.x.max(axis=0)
    Lx, Ly = float(bbmax[0]-bbmin[0]), float(bbmax[1]-bbmin[1])

def get_submesh_by_cell(mesh: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.meshtags, id: int) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.meshtags]:
    assert cell_tags is not None and len(cell_tags.indices) > 0, "Cell tags are not available"
    cells_pg = cell_tags.indices[cell_tags.values == id]
    assert len(cells_pg) > 0, "No cells in physical group"
    tdim = mesh.topology.dim
    return dolfinx.mesh.create_submesh(mesh, tdim, cells_pg)

def compute_scalar(expr: UFLExpr) -> float:
    return dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr))

def get_facet_centroids(mesh: dolfinx.mesh.Mesh, indices: np.ndarray) -> np.ndarray:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, 0)
    f2v = mesh.topology.connectivity(tdim-1, 0)
    centroids = np.array([mesh.geometry.x[f2v.links(f)].mean(axis=0) for f in indices])[:, :2]
    return centroids

def check_periodicity(mesh: dolfinx.mesh.Mesh, ft: dolfinx.mesh.meshtags, fun: dolfinx.fem.Function, right_tag: int, top_tag: int):
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, 0)
    f2v = mesh.topology.connectivity(tdim-1, 0)
    right_vertices = np.concatenate([f2v.links(f) for f in ft.indices[ft.values == right_tag]])
    top_vertices = np.concatenate([f2v.links(f) for f in ft.indices[ft.values == top_tag]])

## Pin a single node to enforce uniqueness
# anchor = np.array([0.5, 0.5])
# i_anchor = np.argmin(np.linalg.norm(V.tabulate_dof_coordinates()[:, :2] - anchor, axis=1))
# bc_anchor = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), np.array([i_anchor], dtype=np.int32), V)
# bcs = [bc_anchor]