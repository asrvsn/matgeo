import numpy as np
from numba import jit
import time, os, math
# from line_profiler import LineProfiler
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

def triangulate_periodic(x: np.ndarray, box: np.ndarray):
        """
        Calculates the periodic triangulation on the set of points x.

        Stores:
            self.n_v = number of vertices (int32)
            self.tris = triangulation of the vertices (nv x 3) matrix.
                Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
            self.vs = coordinates of each vertex; (nv x 2) matrix
            self.v_neighbours = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tris
            self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

        :param x: (nc x 2) matrix with the coordinates of each cell
        """

        # 1. Tile cell positions 9-fold to perform the periodic triangulation
        #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
        #   and the rest are translations

        # TODO: clean-up the grid formation
        grid_x, grid_y = np.mgrid[-1:2,-1:2]
        grid_x[0,0], grid_x[1,1] = grid_x[1,1], grid_x[0,0]
        grid_y[0,0], grid_y[1,1] = grid_y[1,1], grid_y[0,0]
        grid_xy = np.array([grid_x.ravel(), grid_y.ravel()]).T

        grid_xy[:,0] *= box[0]
        grid_xy[:,1] *= box[1]

        y = make_y(x, grid_xy)

        # 2. Perform the triangulation on y
        #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
        #   This triangulation is extracted and saved as tri
        T = Delaunay(y)
        tri = T.simplices

        n_c = x.shape[0]

        # 3. Find triangles with **at least one** cell within the "true" frame 
        # (i.e. with **at least one** "normal cell").
        # Ignore entries with -1, a quirk of the **triangle** package, which 
        # denotes boundary triangles.
        # Generate a mask -- one_in -- that considers such triangles.
        # Save the new triangulation by applying the mask -- new_tri.
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < n_c).any(axis=1)
        tri = tri[one_in]

        # 4. Remove repeats in tri
        #   tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
        tri = remove_repeats(tri,n_c)

        #6. Store outputs
        # self.n_v = tri.shape[0]
        Cents = x[tri]
        vs = circumcenter_periodic(Cents, box)

        #7. Manually calculate the neighbours. See doc_string for conventions.
        v_neighbours = get_neighbours(tri)
        neighbours = vs[v_neighbours]


@jit(nopython=True, cache=True)
def dhdr(rijk):
    """
    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica

    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 0,1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)
    
    return DHDR



@jit(nopython=True, cache=True)
def dhdr_periodic(rijk_, vs, L):
    """
    Same as **dhdr** apart from accounts for periodic triangulation.

    Calculates ∂h_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dhx/drx,dhx/dry),(dhy/drx,dhy/dry))

    These are lifted from Mathematica

    :param rijk_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    L = L.reshape(2,1).T    # reshape to 2D to allow per-dimension operations
    
    rijk = np.empty_like(rijk_)
    for i in range(3):
        rijk[:,i,:] = np.remainder(rijk_[:,i] - vs + L/2, L) - L/2

    DHDR = np.empty(rijk.shape + (2,))
    for i in range(3):
        ax,ay = rijk[:,np.mod(i,3),0],rijk[:,np.mod(i,3),1]
        bx, by = rijk[:, np.mod(i+1,3), 0], rijk[:, np.mod(i+1,3), 1]
        cx, cy = rijk[:, np.mod(i+2,3), 0], rijk[:, np.mod(i+2,3), 1]
        #dhx/drx
        DHDR[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/drx
        DHDR[:, i, 0,1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                    2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhx/dry
        DHDR[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        #dhy/dry
        DHDR[:, i, 1,1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                    (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                        cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

    return DHDR



@jit(nopython=True)
def get_neighbours(tri, neigh=None, Range=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri,dtype=np.int32)*-1

    if Range is None:
        Range = np.arange(n_v)

    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))

    for j in Range: #range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0]) * (tri_compare[:,:,1]==tri_i[k,1]))
                neighb,l = neighb[0],l[0]
                neigh[j,k] = neighb
                neigh[neighb,np.mod(2-l,3)] = j
    
    return neigh



@jit(nopython=True, cache=True)
def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i,Min],tri[i,np.mod(Min+1,3)],tri[i,np.mod(Min+2,3)]
    
    return tri


@jit(nopython=True, cache=True)
def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

    The triangulation is first re-ordered, such that the first cell id 
    referenced is the smallest. Achieved via the function order_tris. 
    (This preserves the internal order -- i.e. CCW)

    Then remove repeated rows via lexsort.

    NB: order of vertices changes via the conventions of lexsort

    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri,n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))

    return sorted_tri[row_mask]


@jit(nopython=True, cache=True)
def circumcenter_periodic(C, L):
    """
    Find the circumcentre (i.e. Voronoi vertex position) of each triangle in the triangulation.
    See Fig. 2b in Barton 2017.
    
    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (2) np.float32 array for width, height
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    L = L.reshape(2,1)    # reshape to 2D to allow per-dimension operations

    ri, rj, rk = C.transpose(1,2,0)
    r_mean = (ri+rj+rk)/3
    disp = r_mean - L/2
    ri, rj, rk = np.mod(ri-disp, L), np.mod(rj-disp, L), np.mod(rk-disp, L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2), dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    vs = np.mod(vs+disp.T, L.T)

    return vs



@jit(nopython=True, cache=True)
def circumcenter(C):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1,2,0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    
    return vs



@jit(nopython=True, cache=True)
def tri_angles(x, tri):
    """
    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0,1,2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0],3,2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (C[:, i_b, 0] - C[:, i_c, 0]) ** 2 + (C[:, i_b, 1] - C[:, i_c, 1]) ** 2
    b2 = (C[:, :, 0] - C[:, i_c, 0] ) ** 2 + (C[:, :, 1] - C[:, i_c, 1]) ** 2
    c2 = (C[:, i_b, 0] - C[:, :, 0] ) ** 2 + (C[:, i_b, 1] - C[:, :, 1]) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    
    return Angles



@jit(nopython=True, cache=True)
def tri_angles_periodic(x, tri, L):
    """
    Same as **tri_angles** apart from accounts for periodic triangulation (i.e. the **L**)

    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0, 1, 2])
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0], 3, 2))
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (np.mod(C[:, i_b, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    b2 = (np.mod(C[:, :, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, :, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    c2 = (np.mod(C[:, i_b, 0] - C[:, :, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, :, 1] + L / 2, L) - L / 2) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)

    return Angles



@jit(nopython=True, cache=True)
def get_k2(tri, v_neighbours):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param v_neighbours: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = v_neighbours[i, k]
            k2 = ((v_neighbours[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    
    return k2s



@jit(nopython=True, cache=True)
def get_k2_boundary(tri, v_neighbours):
    """
    Same as **get_k2** but fills in -1 if the k2 neighbour is undefined (-1)

    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param v_neighbours: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = v_neighbours[i, k]
            if neighbour == -1:
                k2s[i,k] = -1
            else:
                k2 = ((v_neighbours[neighbour] == i) * three).sum()
                k2s[i, k] = k2
    
    return k2s



@jit(nopython=True, cache=True)
def make_y(x, Lgrid_xy):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param Lgrid_xy: (9 x 2) array defining the displacement vectors for each of the 9 images of the tiling
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c = x.shape[0]
    y = np.empty((n_c*9, x.shape[1]))
    for k in range(9):
        y[k*n_c:(k+1)*n_c] = x + Lgrid_xy[k]
    
    return y



@jit(nopython=True, cache=True)
def get_A_periodic(vs, neighbours, Cents, CV_matrix, L, n_c):
    """
    Calculates area of each cell (considering periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param Cents: (nv x 3 x 2) array considering cell centroids of each cell involved in each triangulation
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: self.A saves the areas of each cell
    """
    L = L.reshape(2,1).T    # reshape to 2D to allow per-dimension operations

    AA_mat = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = np.remainder(neighbours[:, np.mod(i + 2, 3)] - Cents[:, i] + L/2, L) - L/2
        Vs = np.remainder(vs - Cents[:, i] + L/2, L) - L/2
        AA_mat[:, i] = 0.5 * (Neighbours[:, 0] * Vs[:, 1] - Neighbours[:, 1] * Vs[:, 0])
    
    A = np.zeros((n_c))
    for i in range(3):
        A += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(AA_mat[:, i])
    
    return A



@jit(nopython=True, cache=True)
def get_A(vs, neighbours, CV_matrix, n_c):
    """
    Calculates area of each cell (considering periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param Cents: (nv x 3 x 2) array considering cell centroids of each cell involved in each triangulation
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: self.A saves the areas of each cell
    """
    AA_mat = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = neighbours[:, np.mod(i + 2, 3)]
        AA_mat[:, i] = 0.5 * (Neighbours[:, 0] * vs[:, 1] - Neighbours[:, 1] * vs[:, 0])
    
    AA_flat = AA_mat.ravel()
    AA_flat[np.isnan(AA_flat)] = 0
    AA_mat = AA_flat.reshape(AA_mat.shape)
    
    A = np.zeros((n_c))
    for i in range(3):
        A += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(AA_mat[:, i])
    
    return A



@jit(nopython=True, cache=True)
def get_P_periodic(vs, neighbours, CV_matrix, L, n_c):
    """
    Finds perimeter of each cell (given periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: P (n_c x 1) np.float32 array of perimeters for each cell
    """
    L = L.reshape(2,1).T    # reshape to 2D to allow per-dimension operations
    
    P_m = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = np.remainder(neighbours[:, i] - vs + L/2, L) - L/2
        P_m[:, i] = np.sqrt((Neighbours[:, 0]) ** 2 + (Neighbours[:, 1]) ** 2)  # * self.boundary_mask

    PP_mat = np.zeros(P_m.shape)
    for i in range(3):
        PP_mat[:, i] = (P_m[:, np.mod(i + 1, 3)] + P_m[:, np.mod(i + 2, 3)]) / 2

    P = np.zeros((n_c))
    for i in range(3):
        P += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(PP_mat[:, i])
    
    return P



@jit(nopython=True, cache=True)
def get_P(vs, neighbours, CV_matrix, n_c):
    """
    Finds perimeter of each cell (given periodic boundary conditions)

    :param vs: (nv x 2) matrix considering coordinates of each vertex
    :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
    :param CV_matrix: cell-vertex binary matrix (n_c x n_v x 3) (where the 3rd dimension prescribes the order)
    :param L: Domain size np.float32
    :param n_c: Number of cells (np.int32)
    :return: P (n_c x 1) np.float32 array of perimeters for each cell
    """
    P_m = np.empty((neighbours.shape[0], neighbours.shape[1]))
    for i in range(3):
        Neighbours = neighbours[:, i] - vs
        P_m[:, i] = np.sqrt((Neighbours[:, 0]) ** 2 + (Neighbours[:, 1]) ** 2)  # * self.boundary_mask

    PP_mat = np.zeros(P_m.shape)
    for i in range(3):
        PP_mat[:, i] = P_m[:, np.mod(i + 2, 3)]

        # PP_mat[:, i] = (P_m[:, np.mod(i + 1, 3)] + P_m[:, np.mod(i + 2, 3)]) / 2

    PP_flat = PP_mat.ravel()
    PP_flat[np.isnan(PP_flat)] = 0
    PP_mat = PP_flat.reshape(PP_mat.shape)

    P = np.zeros((n_c))
    for i in range(3):
        P += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(PP_mat[:, i])
    
    return P



@jit(nopython=True, cache=True)
def roll_forward(x):
    """
    Jitted equivalent to np.roll(x,1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:,2],x[:,:2]))



@jit(nopython=True, cache=True)
def roll_reverse(x):
    """
    Jitted equivalent to np.roll(x,-1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:,1:3],x[:,0]))



@jit(nopython=True, cache=True)
def get_F_periodic(vs, neighbours, tris, CV_matrix, n_v, n_c, L, J_CW, J_CCW, \
                   A, P, X, kappa_A, kappa_P, A0, P0):
    """
    Adapts the computations described in Bi (2016), Appendix A.

    :param vs: Voronoi vertex coordinates.
    :param neighbours: Positions of the three neighbouring vertices (n_v x 3 x 2)
    :param X: Cents
    :param A0: Target cell area.
    :param P0: Target cell perimeter.
    # TODO: Add the rest of the parameters
    """
    L = L.reshape(2,1,1).T  # reshape to 3D to allow per-dimension operations

    # TODO: Rename X -> Cents for consistency with the main program.
    # TODO: Is n_v redundant; always the same length as vs?
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs      # assumed here vs the same length as n_v
        
    # For Voronoi vertex i, neighbours[i,:,:] gives the positions of the three 
    # connected (Voronoi) vertices.
    # roll_forward(), roll_reverse() permutate the neighbours such that 
    # (0,1,2) -> (2,1,0) and (0,1,2) -> (1,2,0), respectively.
    h_jm1 = np.dstack( (roll_forward(neighbours[:,:,0]), roll_forward(neighbours[:,:,1])) )
    h_jp1 = np.dstack( (roll_reverse(neighbours[:,:,0]), roll_reverse(neighbours[:,:,1])) )

    # d[area_j] / d[position] for cell j, Delaunay vertex position. Eqs. (A8), (A9) in Bi.
    dAdh_j = np.mod(h_jp1 - h_jm1 + L/2, L) - L/2
    dAdh_j = np.dstack( (dAdh_j[:,:,1], -dAdh_j[:,:,0]) )   # TODO: check the minus sign

    # d[perimeter_j] / d[positiom]. Eqs. (A8), (A9); Eq. (30) in Barton supplement.
    l_jm1 = np.mod(h_j - h_jm1 + L/2, L) - L/2
    l_jp1 = np.mod(h_j - h_jp1 + L/2, L) - L/2
    l_jm1_norm = np.sqrt( l_jm1[:,:,0]**2 + l_jm1[:,:,1]**2 )
    l_jp1_norm = np.sqrt( l_jp1[:,:,0]**2 + l_jp1[:,:,1]**2 )
    dPdh_j = (l_jm1.T/l_jm1_norm.T + l_jp1.T/l_jp1_norm.T).T
    
    # Derivative of third term in Barton Eq. (1) considering junction lengths.
    dljidh_j = (l_jm1.T * J_CCW.T/l_jm1_norm.T + l_jp1.T * J_CW.T/l_jp1_norm.T).T

    # 3. Cell areas, perimeters for all triangle nodes (n_v x 3)
    vA = A[tris.ravel()].reshape(tris.shape)        # A[tris] ?
    vP = P[tris.ravel()].reshape(tris.shape)        # P[tris] ?

    vA0 = A0[tris.ravel()].reshape(tris.shape)
    vP0 = P0[tris.ravel()].reshape(tris.shape)

    # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and 
    # triangulates with cell j). This last two dims are a Jacobian (2x2) matrix, 
    # defining {x,y} for h and r. (n_v x 3 x 2 x 2)
    DHDR = dhdr_periodic(X, vs, L)  # order is wrt cell i.

    # Note: Barton computes forces for cell centers by looping over cells, but here 
    # the forces are directly on the Voronoi vertices. The position of each Voronoi 
    # vertex depends on the position of three neighboring cells; i.e., each vertex 
    # is enclosed by, or atmost at the boundary of, a triangle formed by three cells. 

    # 5. Calculate the force for each Voronoi vertex wrt the 3 neighbouring cells. 
    # This is essentially decomposing the chain rule of the expression of F for 
    # each cell by vertex. 
    # This is the force contribution for each cell of a given triangle/vertex 
    # (3rd dim). Considering {x,y} components (2nd dim). Within the function, 
    # this calculates (direct and indirect) contributions of each cell wrt each 
    # other cell (i.e. 3x3), then adds them together
    M = np.zeros((2, n_v, 2, 3))

    # Consider the 3 nodes (cell centers) of a triangle: Taking each node as the
    # cell for which the force is to be computed (index i) in turn, sum up the
    # contributions due to cell i vertices (when j=i), and the contributions due
    # to the (overlapping) seneighbor cell vertices (when j!=i).
    # See Barton Eq. (7) algorithm description - which is explicitly over all
    # associated vertices, but gives the idea of how the cell vertices plus 
    # neighboring cell vertices are considered.
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):   # in 2D
                # First two terms (area and perimenter) follow the formulation
                # of Bi: Barton uses a different expression for cell area, and
                # the perimeter in Barton ignores P0 (TODO: check if should remove 
                # it here). Third term (junction) follows Barton.
                M[:, :, Fdim, i] += ( kappa_A * (vA[:,j] - vA0[:,j]) * dAdh_j[:, j].T +
                                      kappa_P * (vP[:,j] - vP0[:,j]) * dPdh_j[:, j].T +
                                      dljidh_j[:,j].T ) * DHDR[:, i, Fdim].T
    
    M = M[0] + M[1]

    # 6. Compile force components wrt cells by using the cell-to-vertex connection matrix.
    # Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt cell i}
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        dEdr += np.asfortranarray(CV_matrix[:,:,i]) @ np.asfortranarray(M[:,:,i])
    
    F = -dEdr

    return F



@jit(nopython=True, cache=True)
def get_area_cost_grad_periodic(xs, neighbours, tris, L, A, kappa_A, A0):
    """
    Like get_F_periodic, but simplify calculations without line tension.

    :param xs: seed points
    :param vs: Voronoi vertex coordinates (of length n_v)
    :param neighbours: Positions of the three neighbouring vertices (n_v x 3 x 2)
    :param tris: indices of the three vertices forming each Delaunay triangle (n_v x 3)
    :param Cents: Cents
    :param A0: Target cell area.
    # TODO: Add the rest of the parameters
    """
    n_c = xs.shape[0]
    Cents = xs[tris]
    vs = circumcenter_periodic(Cents, L)
    
    L = L.reshape(2,1,1).T  # reshape to 3D to allow per-dimension operations

    n_v = vs.shape[0]
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs      # assumed here vs the same length as n_v
        
    # For Voronoi vertex i, neighbours[i,:,:] gives the positions of the three 
    # connected (Voronoi) vertices.
    # roll_forward(), roll_reverse() permutate the neighbours such that 
    # (0,1,2) -> (2,1,0) and (0,1,2) -> (1,2,0), respectively.
    h_jm1 = np.dstack( (roll_forward(neighbours[:,:,0]), roll_forward(neighbours[:,:,1])) )
    h_jp1 = np.dstack( (roll_reverse(neighbours[:,:,0]), roll_reverse(neighbours[:,:,1])) )

    # d[area_j] / d[position] for cell j, Delaunay vertex position. Eqs. (A8), (A9) in Bi.
    dAdh_j = np.mod(h_jp1 - h_jm1 + L/2, L) - L/2
    dAdh_j = np.dstack( (dAdh_j[:,:,1], -dAdh_j[:,:,0]) )   # TODO: check the minus sign

    # 3. Cell areas for all triangle nodes (n_v x 3)
    vA = A[tris]
    vA0 = A0[tris]

    # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and 
    # triangulates with cell j). This last two dims are a Jacobian (2x2) matrix, 
    # defining {x,y} for h and r. (n_v x 3 x 2 x 2)
    DHDR = dhdr_periodic(Cents, vs, L)  # order is wrt cell i.

    # Note: Barton computes forces for cell centers by looping over cells, but here 
    # the forces are directly on the Voronoi vertices. The position of each Voronoi 
    # vertex depends on the position of three neighboring cells; i.e., each vertex 
    # is enclosed by, or atmost at the boundary of, a triangle formed by three cells. 

    # 5. Calculate the force for each Voronoi vertex wrt the 3 neighbouring cells. 
    # This is essentially decomposing the chain rule of the expression of F for 
    # each cell by vertex. 
    # This is the force contribution for each cell of a given triangle/vertex 
    # (3rd dim). Considering {x,y} components (2nd dim). Within the function, 
    # this calculates (direct and indirect) contributions of each cell wrt each 
    # other cell (i.e. 3x3), then adds them together
    M = np.zeros((2, n_v, 2, 3))

    # Consider the 3 nodes (cell centers) of a triangle: Taking each node as the
    # cell for which the force is to be computed (index i) in turn, sum up the
    # contributions due to cell i vertices (when j=i), and the contributions due
    # to the (overlapping) seneighbor cell vertices (when j!=i).
    # See Barton Eq. (7) algorithm description - which is explicitly over all
    # associated vertices, but gives the idea of how the cell vertices plus 
    # neighboring cell vertices are considered.
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):   # in 2D
                # First two terms (area and perimenter) follow the formulation
                # of Bi: Barton uses a different expression for cell area, and
                # the perimeter in Barton ignores P0 (TODO: check if should remove 
                # it here). Third term (junction) follows Barton.
                M[:, :, Fdim, i] += ( kappa_A * (vA[:,j] - vA0[:,j]) * dAdh_j[:, j].T
                                    ) * DHDR[:, i, Fdim].T
    
    M = M[0] + M[1]

    # 6. Compile force components wrt cells by using the cell-to-vertex connection matrix.
    # Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt cell i}
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        CV_matrix_i = np.zeros((n_c, n_v))
        CV_matrix_i[tris[:, i], np.arange(n_v)] = 1
        dEdr += np.asfortranarray(CV_matrix_i) @ np.asfortranarray(M[:,:,i])
    
    return dEdr


@jit(nopython=True, cache=True)
def get_F(vs, neighbours, tris, CV_matrix, n_v, n_c, L, J_CW, J_CCW, A, P, X, \
          kappa_A, kappa_P, A0, P0, n_C, kappa_B, l_b0):
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jm1 = np.dstack((roll_forward(neighbours[:,:,0]),roll_forward(neighbours[:,:,1])))
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))

    dAdh_j = h_jp1 - h_jm1          # d[Area] / d[position] ? 
    dAdh_j = np.dstack((dAdh_j[:,:,1],-dAdh_j[:,:,0]))

    l_jm1 = h_j - h_jm1
    l_jp1 = h_j - h_jp1
    l_jm1_norm, l_jp1_norm = np.sqrt(l_jm1[:,:,0] ** 2 + l_jm1[:,:,1] ** 2), np.sqrt(l_jp1[:,:,0] ** 2 +  l_jp1[:,:,1] ** 2)
    dPdh_j = (l_jm1.T/l_jm1_norm.T + l_jp1.T/l_jp1_norm.T).T

    dljidh_j = (l_jm1.T * J_CCW.T/l_jm1_norm.T + l_jp1.T * J_CW.T/l_jp1_norm.T).T

    ## 3. Find areas and perimeters of the cells and restructure data wrt. the triangulation
    # vA = A[tris.ravel()].reshape(tris.shape)
    # vP = P[tris.ravel()].reshape(tris.shape)

    real_cell = np.zeros(n_c)
    real_cell[:n_C] = 1

    vdA = ((A-A0)*real_cell)[tris.ravel()].reshape(tris.shape)
    vdP = ((P-P0)*real_cell)[tris.ravel()].reshape(tris.shape)

    # 4. Calculate ∂h/∂r. This is for cell i (which may or may not be cell j, and triangulates with cell j)
    # This last two dims are a Jacobinan (2x2) matrix, defining {x,y} for h and r. See function description for details
    DHDR = dhdr(X)  # order is wrt cell i

    vRC = real_cell[tris.ravel()].reshape(tris.shape)
    bEdge = vRC*roll_forward(1-vRC) + (1-vRC)*roll_forward(vRC) + vRC*roll_reverse(1-vRC) + (1-vRC)*roll_reverse(vRC)

    # 5. Now calculate the force component for each vertex, with respect to the 3 neighbouring cells
    #   This is essentially decomposing the chain rule of the expression of F for each cell by vertex
    #   M_sum is a (nv,2,3) matrix. This is the force contribution for each cell of a given triangle/vertex (3rd dim). Considering {x,y} components (2nd dim)
    #       Within the function, this calculates (direct and indirect) contributions of each cell wrt each other cell (i.e. 3x3), then adds them together
    M = np.zeros((2, n_v, 2, 3))
    for i in range(3):
        for j in range(3):
            for Fdim in range(2):
                M[:, :, Fdim, i] += DHDR[:, i, Fdim].T * \
                                    (kappa_A * vdA[:,j] * dAdh_j[:, j].T
                                     + kappa_P * vdP[:,j] * dPdh_j[:, j].T
                                     + vRC[:,j]*dljidh_j[:,j].T
                                     # + dljidh_j[:, j].T
                                     + kappa_B*bEdge[:,j]*(l_jp1_norm[:,j] - l_b0)*dPdh_j[:,j].T)
    M = M[0] + M[1]

    M_flat = M.ravel()
    M_flat[np.isnan(M_flat)] = 0
    M = M_flat.reshape(M.shape)

    # 6. Compile force components wrt. cells by using the cell-to-vertex connection matrix.
    #       Force on cell_i = SUM_{vertices of cell i} {forces at each vertex wrt. cell i}
    dEdr = np.zeros((n_c, 2))
    for i in range(3):
        dEdr += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(M[:, :, i])
    F = -dEdr

    return F



@jit(nopython=True, cache=True)
def weak_repulsion(Cents, a, k, CV_matrix, n_c, L):
    """
    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    And is performed respecting periodic boudnary conditions (system size = L)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :param L: Domain size (2) np.float32 array for width, height
    :return: F_soft
    """
    L = L.reshape(2,1,1)    # reshape to 3D to allow per-dimension operations

    CCW = np.dstack((roll_reverse(Cents[:,:,0]), roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    displacement = np.mod((Cents - CCW).T + L/2, L) - L/2
    rij = np.sqrt(displacement[0,:,:]**2 + displacement[1,:,:]**2)
    norm_disp = (displacement / rij).T
    V_soft_mag = -k * (rij.T - 2*a) * (rij.T < 2*a)
    V_soft_CCW = (V_soft_mag.T * norm_disp.T).T
    V_soft_CW = -(roll_forward(V_soft_mag).T * norm_disp.T).T
    V_soft = V_soft_CW + V_soft_CCW

    F_soft = np.zeros((n_c, 2))
    for i in range(3):
        F_soft += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(V_soft[:, i])
    
    return F_soft



@jit(nopython=True, cache=True)
def get_l_interface(n_v, n_c, neighbours, vs, CV_matrix, L):
    """
    Get the length of the interface between each pair of cells.

    LI[i,j] = length of interface between cell i and j = L[j,i] (if using periodic triangulation)

    :param n_v: Number of vertices (**np.int64**)
    :param n_c: Number of cells (**np.int64**
    :param neighbours: Position of the three neighbouring vertices (n_v x 3 x 2)
    :param vs: Positions of vertices (n_v x 3)
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :param L: Domain size (**np.float32**)
    :return:
    """
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))
    l = np.mod(h_j - h_jp1 + L/2,L) - L/2
    l = np.sqrt(l[:,:,0]**2 + l[:,:,1]**2)
    LI = np.zeros((n_c,n_c),dtype=np.float32)
    for i in range(3):
        LI+= np.asfortranarray(l[:,i]*CV_matrix[:,:,i])@np.asfortranarray(CV_matrix[:,:,np.mod(i+2,3)].T)
    
    return LI



def get_l_interface_boundary(n_v, n_c, neighbours, vs, CV_matrix):
    """
    Same as **get_l_interface** but accounts for boundary cells

    Get the length of the interface between each pair of cells.

    LI[i,j] = length of interface between cell i and j = L[j,i]

    Note: the difference lies in the making the LI matrix symmetric.

    :param n_v: Number of vertices (**np.int64**)
    :param n_c: Number of cells (**np.int64**
    :param neighbours: Position of the three neighbouring vertices (n_v x 3 x 2)
    :param vs: Positions of vertices (n_v x 3)
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jp1 = np.dstack((roll_reverse(neighbours[:,:,0]),roll_reverse(neighbours[:,:,1])))
    l = h_j - h_jp1
    l = np.sqrt(l[:,:,0]**2 + l[:,:,1]**2)
    LI = np.zeros((n_c,n_c),dtype=np.float32)
    for i in range(3):
        LI+= np.asfortranarray(l[:,i]*CV_matrix[:,:,i])@np.asfortranarray(CV_matrix[:,:,np.mod(i+2,3)].T)
    LI = np.dstack((LI,LI.T)).max(axis=2)

    return LI



@jit(nopython=True, cache=True)
def weak_repulsion_boundary(Cents, a, k, CV_matrix, n_c, n_C):
    """
    Identical to **weak_repulsion** apart from without periodic boundary conditions

    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :return: F_soft
    """
    CCW = np.dstack((roll_reverse(Cents[:,:,0]),roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CCW_displacement = Cents - CCW
    rij = np.sqrt(CCW_displacement[:,:,0]**2 + CCW_displacement[:,:,1]**2)
    norm_disp = (CCW_displacement.T/rij.T).T
    V_soft_mag = -k*(rij - 2*a)*(rij<2*a)
    V_soft_CCW = (V_soft_mag.T*norm_disp.T).T
    V_soft_CW = -(roll_forward(V_soft_mag).T*norm_disp.T).T
    V_soft = V_soft_CW + V_soft_CCW
    F_soft = np.zeros((n_c, 2))
    for i in range(3):
        F_soft += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(V_soft[:, i])
    F_soft[n_C:] = 0

    return F_soft



@jit(nopython=True, cache=True)
def boundary_tension(Gamma_bound, n_C, n_c, Cents, CV_matrix):
    boundary_cells = np.zeros(n_c)
    boundary_cells[n_C:] = 1
    CCW = np.dstack((roll_reverse(Cents[:,:,0]),roll_reverse(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CW = np.dstack((roll_forward(Cents[:,:,0]),roll_forward(Cents[:,:,1])))#np.column_stack((Cents[:,1:3],Cents[:,0].reshape(-1,1,2)))
    CCW_displacement = Cents - CCW
    CW_displacement = Cents - CW

    forces = -CW_displacement - CCW_displacement
    F_boundary = np.zeros((n_c, 2))
    for i in range(3):
        F_boundary += np.asfortranarray(CV_matrix[:, :, i])@np.asfortranarray(forces[:,i])
    F_boundary[:n_C] = 0

    return Gamma_bound*F_boundary



@jit(nopython=True, cache=True)
def get_C(n_c, CV_matrix):
    """
    Generates a cell-cell interaction matrix (binary) (n_c x n_c).

    If entry C[i,j] is 1, means that cell j is the CW neighbour of cell i in one of the triangles of the triangulation

    Note: if the triangulation is not periodic, this directionality will result in asymmetric entries of rows/cols
    associated with boundary cells. To generate a symmetric interaction matrix, perform (C + C.T)!=0

    :param n_c: Number of cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    C = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(3):
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 2, 3)].T)
    C = (C != 0).astype(np.int32)

    return C



@jit(nopython=True, cache=True)
def get_C_boundary(n_c, CV_matrix):
    """
    Generates a cell-cell interaction matrix (binary) (n_c x n_c).

    If entry C[i,j] is 1, means that cell j is the CW neighbour of cell i in one of the triangles of the triangulation

    Note: if the triangulation is not periodic, this directionality will result in asymmetric entries of rows/cols
    associated with boundary cells. To generate a symmetric interaction matrix, perform (C + C.T)!=0

    :param n_c: Number of cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :return:
    """
    C = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(3):
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 2, 3)].T)
        C += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(CV_matrix[:, :, np.mod(i + 1, 3)].T)
    C = (C != 0).astype(np.int32)

    return C



@jit(nopython=True, cache=True)
def get_F_bend(n_c, CV_matrix, n_C, x, zeta):
    """
    Get the spatial differential of the bending energy i.e. the bending force -- F_bend

    If E_bend = Sum_i{zeta_i * cos(theta_i))
    where cos(theta_i) = (r_{ji}•r_{ki})/(|r_{ji}||r_{ki}|)

    Then F_bend|_{cell_i} = - \partial E_bend / \partial r_i

    Relies on the function **dcosthetadr** which specifies:
        \partial cos(theta_i) / \partial r_i
        \partial cos(theta_i) / \partial r_j
        \partial cos(theta_i) / \partial r_k

    :param n_c: Number of cells including boundary cells **np.int64**
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :param n_C: Number of cells exclusing boundary cells **np.int64**
    :param x: Cell centroids (n_c x 2)
    :param zeta: Coefficient of bending energy **np.float64**
    :return:
    """
    C_b = np.zeros((n_c-n_C, n_c-n_C), dtype=np.float64)
    for i in range(3):
        C_b += np.asfortranarray(CV_matrix[n_C:, :, i]) @ np.asfortranarray(CV_matrix[n_C:, :, np.mod(i + 2, 3)].T)

    x_i = x[n_C:]
    x_j = np.asfortranarray(C_b)@x_i
    x_k = np.asfortranarray(C_b.T)@x_i

    dC_ri, dC_rj,dC_rk = dcosthetadr(x_i,x_j,x_k)

    F_b = zeta *(dC_ri
                  + np.asfortranarray(C_b) @ np.asfortranarray(dC_rj)
                  + np.asfortranarray(C_b.T) @ np.asfortranarray(dC_rk))

    F_bend = np.zeros((n_c, 2))
    F_bend[n_C:] = F_b

    return F_bend



@jit(nopython=True,cache=True)
def dcosthetadr(ri, rj, rk):
    """
    If cos(theta_i) = (r_{ji}•r_{ki})/(|r_{ji}||r_{ki}|)

    then this function calculates:
        \partial cos(theta_i) / \partial r_i (denoted in short hand dC_ri)
        \partial cos(theta_i) / \partial r_j (denoted in short hand dC_rj)
        \partial cos(theta_i) / \partial r_k (denoted in short hand dC_rk)

    :param ri: Array of positions of boundary cell i (n_c - n_C x 2)
    :param rj: Array positions of neighbours of i (j) (n_c - n_C x 2)
    :param rk: Array of positions of the other neighbour of i (k) (n_c - n_C x 2)
    :return: dC_ri, dC_rj,dC_rk
    """
    rij = ri-rj
    rjk = rj - rk
    rki = rk - ri
    nij = np.sqrt(rij[:,0]** 2 + rij[:,1]**2)
    njk = np.sqrt(rjk[:,0]** 2 + rjk[:,1]**2)
    nki = np.sqrt(rki[:,0]** 2 + rki[:,1]**2)

    dC_ri = (1/(2*nij*nki))*((rij.T/nij**2)*(nki**2 - njk**2 - nki**2)
                             - (rki.T/nki**2)*(nij**2 - njk**2 - nki**2))
    dC_rj = (1/(2*nij**3 * nki)*((rij.T * (njk**2 - nki**2)) + (nij**2 * (rjk.T - rki.T))))
    dC_rk = (1/(2*nki**3 * nij)* ((rki.T * (-njk**2 + nij**2) + (nki**2 * (rij.T - rjk.T)))))

    return dC_ri.T, dC_rj.T,dC_rk.T
