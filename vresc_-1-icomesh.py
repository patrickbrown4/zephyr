"""
Large sections of this code are copied verbatim from
http://sinestesia.co/blog/tutorials/python-icospheres/
"""

###############
### IMPORTS ###
import numpy as np
import pandas as pd
import os, sys, math
import geopandas as gpd
import shapely
from tqdm import tqdm, trange

### Only need zephyr for projpath
import zephyr
projpath = zephyr.settings.projpath

#######################
### ARGUMENT INPUTS ###
import argparse
parser = argparse.ArgumentParser(description='Create US icosahedral mesh')
parser.add_argument('subdiv', type=int, default=9, help='number of edge subdivisions')
### Parse arguments
args = parser.parse_args()
subdiv = args.subdiv

#################
### CONSTANTS ###

### Golden ratio
PHI = (1 + math.sqrt(5)) / 2

#################
### FUNCTIONS ###

def rot(vector, axis, angle):
    """
    Rotate vector about axis by angle.
    Inputs
    ------
    vector: np.array with shape == (3,)
    axis: str in ['x', 'y', 'z']
    angle: angle in degrees
    """
    assert axis in ['x', 'y', 'z'], "axis must be in ['x', 'y', 'z']"
    if axis.lower() == 'x':
        rotation = np.array([
            [1, 0,                                0                              ],
            [0, math.cos(math.pi / 180 * angle), -math.sin(math.pi / 180 * angle)],
            [0, math.sin(math.pi / 180 * angle),  math.cos(math.pi / 180 * angle)],
        ])
    elif axis.lower() == 'y':
        rotation = np.array([
            [ math.cos(math.pi / 180 * angle), 0, math.sin(math.pi / 180 * angle)],
            [0,                                1, 0                              ],
            [-math.sin(math.pi / 180 * angle), 0, math.cos(math.pi / 180 * angle)],
        ])
    elif axis.lower() == 'z':
        rotation = np.array([
            [math.cos(math.pi / 180 * angle), -math.sin(math.pi / 180 * angle), 0],
            [math.sin(math.pi / 180 * angle),  math.cos(math.pi / 180 * angle), 0],
            [0,                               0,                                1],
        ])
    return np.dot(rotation, vector)

### Scale a vector to the unit sphere
def vertex(x, y, z, scale=1):
    """
    Scale a vector to the unit sphere.
    Return vertex coordinates fixed to the unit sphere.
    """
    length = math.sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

def icomesh(subdiv=0, rotax1='x', rotang1=0., rotax2='x', rotang2=0.):
    """
    Outputs
    -------
    tuple of (verts, faces)
    verts: list of vertex vectors np.arrays
    faces: list of lists identifying faces by vertex trios
    """
    ### Make the vertices of the base icosahedron
    verts = [vertex(-1,  PHI, 0),
             vertex( 1,  PHI, 0),
             vertex(-1, -PHI, 0),
             vertex( 1, -PHI, 0),
             vertex(0, -1, PHI),
             vertex(0,  1, PHI),
             vertex(0, -1, -PHI),
             vertex(0,  1, -PHI),
             vertex( PHI, 0, -1),
             vertex( PHI, 0,  1),
             vertex(-PHI, 0, -1),
             vertex(-PHI, 0,  1)]
    
    ### Rotate the base icosahedron
    verts = [rot(vert, rotax1, rotang1) for vert in verts]
    verts = [rot(vert, rotax2, rotang2) for vert in verts]

    ### Identify faces
    faces = [## 5 faces around point 0
             [0, 11, 5],
             [0, 5, 1],
             [0, 1, 7],
             [0, 7, 10],
             [0, 10, 11],
             ## Adjacent faces
             [1, 5, 9],
             [5, 11, 4],
             [11, 10, 2],
             [10, 7, 6],
             [7, 1, 8],
             ## 5 faces around 3
             [3, 9, 4],
             [3, 4, 2],
             [3, 2, 6],
             [3, 6, 8],
             [3, 8, 9],
             ## Adjacent faces
             [4, 9, 5],
             [2, 4, 11],
             [6, 2, 10],
             [8, 6, 7],
             [9, 8, 1]]
    
    ### Make and subdivide edges
    ## Cache for points that have been cut
    middle_point_cache = {}
    ## Cutting function
    def middle_point(point_1, point_2):
        """
        Find the middle point between each vertex pair
        and project to the unit sphere
        """
        ## Check if edge is already in cache to avoid duplicates
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in middle_point_cache:
            return middle_point_cache[key]

        ## Cut edge if it is not in cache
        vert_1 = verts[point_1]
        vert_2 = verts[point_2]
        middle = [sum(i)/2 for i in zip(vert_1, vert_2)]

        verts.append(vertex(*middle))

        index = len(verts) - 1
        middle_point_cache[key] = index

        return index

    ## Loop and make the subdivisions
    for i in trange(subdiv, desc='subdivisions:'):
        faces_subdiv = []

        for tri in faces:
            v1 = middle_point(tri[0], tri[1])
            v2 = middle_point(tri[1], tri[2])
            v3 = middle_point(tri[2], tri[0])

            faces_subdiv.append([tri[0], v1, v3])
            faces_subdiv.append([tri[1], v2, v1])
            faces_subdiv.append([tri[2], v3, v2])
            faces_subdiv.append([v1, v2, v3])

        faces = faces_subdiv
    
    ### Return results
    return verts, faces

def unitify(angle):
    """
    Shift any angle to (-180,180]
    """
    if angle > 180:
        angle -= 360
        return unitify(angle)
    if angle <= -180:
        angle += 360
        return unitify(angle)
    return angle

def lonlatify(vector):
    """
    Return (lat, lon) of unit vector in degrees.
    """
    x, y, z = vector[0], vector[1], vector[2]
    ### Latitude: Straightforward
    lat = 180 / math.pi * math.asin(z)
    ### Longitude: Account for edge cases
    ## Normal cases
    if   (x > 0.) and (y != 0.):
        lon = 180 / math.pi * math.atan(y/x)
    elif (x < 0.) and (y > 0.):
        lon = 180 + 180 / math.pi * math.atan(y/x)
    elif (x < 0.) and (y < 0.):
        lon = -180 + 180 / math.pi * math.atan(y/x)
    ## Edge cases
    elif x > 0. and y == 0.:
        lon = 0.
    elif x < 0. and y == 0.:
        lon = 180
    elif x == 0. and y > 0.:
        lon = 90.
    elif x == 0. and y < 0.:
        lon = -90.
    elif x == 0. and y == 0.:
        lon = 0.
    ## Catch pathological cases
    else:
        raise Exception('Extra special lon case.')
    ### Return results
    return unitify(lon), lat

#################
### PROCEDURE ###

###### Get the US shapefile
dfmap = gpd.read_file(
    os.path.join(projpath,'in','Maps','Census','cb_2016_us_nation_5m','cb_2016_us_nation_5m.shp')
)
biggest_poly = max([
    len(dfmap.geometry[0][i].__str__()) 
    for i in range(len(dfmap.geometry[0]))])
us_contig_index = [
    len(dfmap.geometry[0][i].__str__()) 
    for i in range(len(dfmap.geometry[0]))].index(biggest_poly)
usa_poly = dfmap.loc[0,'geometry'][us_contig_index]
usa_centroid_lon = usa_poly.centroid.x
usa_centroid_lat = usa_poly.centroid.y

###### Make the icosahedral mesh
#### Rotate the icosahedron to give roughly uniform spacing over US
# ### Example - no rotation
# rotax1, rotang1 = 'x', 0.
# rotax2, rotang2 = 'x', 0.
### Actual rotation used
rotax1, rotang1 = 'x', -180 / math.pi * math.atan(1 / PHI) + 90 - usa_centroid_lat - 11
rotax2, rotang2 = 'z', 90 + usa_centroid_lon

### Get the vertices and faces for the base icosahedron
verts_0, faces_0 = icomesh(0, rotax1, rotang1, rotax2, rotang2)
lons_0 = [lonlatify(vert)[0] for vert in verts_0]
lats_0 = [lonlatify(vert)[1] for vert in verts_0]

### Get the vertices and faces for the subdivided points
verts, faces = icomesh(subdiv, rotax1, rotang1, rotax2, rotang2)
lons = [lonlatify(vert)[0] for vert in verts]
lats = [lonlatify(vert)[1] for vert in verts]
lonlats = (lons, lats)

###### Merge with US shapefile
lonlats = pd.DataFrame(data=list(zip(lons, lats)), columns=['lon', 'lat'], dtype=float)

### USA:
lonmin = usa_poly.bounds[0] - 3
latmin = usa_poly.bounds[1] - 3
lonmax = usa_poly.bounds[2] + 3
latmax = usa_poly.bounds[3] + 3

points = lonlats[
    (lonlats['lon'] <= lonmax)
    & (lonlats['lon'] >= lonmin)
    & (lonlats['lat'] <= latmax)
    & (lonlats['lat'] >= latmin)
].values

### Mask points to US boundary
usapoints = []
for i in trange(len(points), desc='mask to USA boundary'):
    point = shapely.geometry.Point(points[i])
    if point.within(dfmap.geometry[0]):
        usapoints.append(points[i])

### Write it
savename = 'usa-points-icomesh-x[-atan(invPHI)+90-lat-11]-z[90+lon]-{}subdiv'.format(subdiv)

pd.DataFrame(
    usapoints, columns=['lon', 'lat']
).to_csv(os.path.join('io','{}.csv').format(savename), index=False)
