'''
Utilities for dolfinx mesh
'''
import numpy as np
import dolfinx.mesh
import dolfinx.plot
import pyvista

def get_exterior_tags(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.meshtags:
    '''
    Get the tags for the inclusions in the mesh
    '''
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    vals = np.ones(len(facets), dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim-1, facets, vals)

def visualize_exterior_tags(mesh: dolfinx.mesh.Mesh):
    ft = get_exterior_tags(mesh)
    tdim = mesh.topology.dim
    # Interior
    topo, types, geom = dolfinx.plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topo, types, geom)
    # Exterior
    f_topo, f_types, f_geom = dolfinx.plot.vtk_mesh(mesh, tdim-1)
    f_grid = pyvista.UnstructuredGrid(f_topo, f_types, f_geom)
    tags = np.full(f_grid.n_cells, -1, dtype=np.int32)
    tags[ft.indices] = ft.values
    f_grid.cell_data['FacetTags'] = tags
    # Plot
    p = pyvista.Plotter(window_size=(1000, 1000))
    p.add_mesh(grid, show_edges=True, opacity=0.2)
    p.add_mesh(f_grid, scalars='FacetTags', line_width=2, render_lines_as_tubes=True, show_scalar_bar=True, lighting=False)
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
    # if warp:
    #     warped = grid.warp_by_scalar()
    #     p.add_mesh(warped)
    p.show()