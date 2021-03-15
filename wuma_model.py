"""Writes out the cmod file."""


from enum import Enum
import struct
from typing import BinaryIO, List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from scipy.linalg import norm
from scipy.optimize import approx_fprime, brentq, newton

CEL_MODEL_HEADER_BINARY = b"#celmodel_binary"


# from modelfile.h
class ModelFileToken(Enum):
    MATERIAL = 1001
    END_MATERIAL = 1002
    DIFFUSE = 1003
    SPECULAR = 1004
    SPECULAR_POWER = 1005
    OPACITY = 1006
    TEXTURE = 1007
    MESH = 1009
    END_MESH = 1010
    VERTEX_DESC = 1011
    END_VERTEX_DESC = 1012
    VERTICES = 1013
    EMISSIVE = 1014
    BLEND = 1015


# from modelfile.h
class ModelFileType(Enum):
    FLOAT1 = 1
    FLOAT2 = 2
    FLOAT3 = 3
    FLOAT4 = 4
    STRING = 5
    UINT32 = 6
    COLOR = 7


# from mesh.h
class VertexAttributeSemantic(Enum):
    POSITION = 0
    COLOR0 = 1
    COLOR1 = 2
    NORMAL = 3
    TANGENT = 4
    TEXTURE0 = 5
    TEXTURE1 = 6
    TEXTURE2 = 7
    TEXTURE3 = 8
    POINT_SIZE = 9
    NEXT_POSITION = 10
    SCALE_FACTOR = 11


# from mesh.h
class VertexAttributeFormat(Enum):
    FLOAT1 = 0
    FLOAT2 = 1
    FLOAT3 = 2
    FLOAT4 = 3
    UBYTE4 = 4


# from mesh.h
class PrimitiveGroupType(Enum):
    TRI_LIST = 0
    TRI_STRIP = 1
    TRI_FAN = 2
    LINE_LIST = 3
    LINE_STRIP = 4
    POINT_LIST = 5
    SPRITE_LIST = 6


S_INT16 = struct.Struct('<h')
S_UINT32 = struct.Struct('<I')
S_COLOR = struct.Struct('<h3f')
S_FLOAT1 = struct.Struct('<hf')
S_TEXTURE = struct.Struct('<4h')
S_VERTEX_DESC = struct.Struct('<2h')
S_VERTEX = struct.Struct('<8f')
S_GROUP = struct.Struct('<h2I')


# Pre-compute cos(theta), sin(theta). Due to symmetry we only need to compute one quadrant in yz.
def _init_angular(samples: int) -> Tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(0.0, np.pi/2, samples)
    cos_vals = np.cos(thetas)
    sin_vals = np.sin(thetas)
    # use exact values at endpoints
    cos_vals[0] = 1
    sin_vals[0] = 0
    cos_vals[-1] = 0
    sin_vals[-1] = 1
    return cos_vals, sin_vals


X_SAMPLES = 72
THETA_SAMPLES = 12
COS_VALS, SIN_VALS = _init_angular(THETA_SAMPLES)
U_VALS = np.linspace(0, 0.25, THETA_SAMPLES)


# Roche potential with mass 1 at (0,0,0), mass q at (1,0,0)
def roche(xyz: ArrayLike, q: float) -> float:
    """Dimensionless Roche potential along x-axis."""
    r1 = norm(xyz)
    r2 = norm([xyz[0]-1, xyz[1], xyz[2]])
    return 2/((1+q)*r1) + 2*q/((1+q)*r2) + (xyz[0]-q/(1+q))**2 + xyz[1]*xyz[1]


# Lagrange point finding (configuration as for Roche potential above)
def _l1(x: float, q: float) -> float:
    return q/((1+q)*(1-x)**2) - 1/((1+q)*x**2) + x-q/(1+q)


# first derivative of _l1, for use in Newton's method
def _l1prime(x: float, q: float) -> float:
    return 2*q/((1+q)*(1-x)**3) + 2/((1+q)*x**3) + 1


def _l2(x: float, q: float) -> float:
    return -q/((1+q)*(x-1)**2) - 1/((1+q)*x**2) + x-q/(1+q)


def _l2prime(x: float, q: float) -> float:
    return 2*q/((1+q)*(x-1)**3) + 2/((1+q)*x**3) + 1


def _l3(x: float, q: float) -> float:
    return q/((1+q)*(1-x)**2) + 1/((1+q)*x**2) + x-q/(1+q)


def _l3prime(x: float, q: float) -> float:
    return 2*q/((1+q)*(1-x)**3) - 2/((1+q)*x**3) + 1


def lagrange_points(q: float) -> Tuple[float, float, float]:
    """Computes the locations of the colinear Lagrange points."""
    # use Newton to avoid singularities at interval endpoints
    l1 = newton(_l1, 0.5, fprime=_l1prime, args=(q,))
    l2 = newton(_l2, 2-l1, fprime=_l2prime, args=(q,))
    l3 = newton(_l3, 1-l2, fprime=_l3prime, args=(q,))
    return l1, l2, l3


# determine x-axis extents of common envelope
def _extent_max(phi: float, l2: float, q: float) -> float:
    # find interval containing value by stepping inwards from L2 to secondary
    outer = l2
    inner = (l2+1)/2
    ph = roche([inner, 0, 0], q)
    while ph < phi:
        outer = inner
        inner = (inner+1)/2
        ph = roche([inner, 0, 0], q)
    # find value in interval
    return brentq(lambda x: roche([x, 0, 0], q)-phi, inner, outer)


def _extent_min(phi: float, l3: float, q: float) -> float:
    # find interval containing value by stepping inwards from L3 to primary
    outer = l3
    inner = l3/2
    ph = roche([inner, 0, 0], q)
    while ph < phi:
        outer = inner
        inner /= 2
        ph = roche([inner, 0, 0], q)
    return brentq(lambda x: roche([x, 0, 0], q)-phi, outer, inner)


# find radial distance from x-axis along a ray at givn x-coordinate and theta in yz-plane
def _surface(phi: float, x: float, theta_i: int, outer: float, q: float) -> float:
    inner = outer/2
    ct = COS_VALS[theta_i]
    st = SIN_VALS[theta_i]
    ph = roche([x, inner*ct, inner*st], q)
    while ph < phi:
        outer = inner
        inner /= 2
        ph = roche([x, inner*ct, inner*st], q)
    return brentq(lambda r: roche([x, r*ct, r*st], q)-phi, inner, outer)


class Vertex:
    """Mesh vertex."""
    def __init__(self, position: np.ndarray, normal: np.ndarray, uv: np.ndarray):
        self.position = position
        self.normal = normal
        self.uv = uv

    def __repr__(self):
        return f'Vertex({self.position!r}, {self.normal!r}, {self.uv!r})'


class MeshGroup:
    """Mesh group."""
    def __init__(self, primitive: PrimitiveGroupType, material: int, indices: List[int]):
        self.primitive = primitive
        self.material = material
        self.indices = indices

# pre-build groups
def _build_groups() -> List[MeshGroup]:
    ring_size = 4*THETA_SAMPLES - 3
    num_vertices = 2 + ring_size*X_SAMPLES

    # x_min cap
    groups = [MeshGroup(
        PrimitiveGroupType.TRI_FAN,
        0,
        list(range(ring_size+1)))
    ]

    # triangle strip goes ring 1 point 1, ring 2 point 1, ring 1 point 2, ring 2 point 2, etc.
    offsets = np.empty(ring_size*2, dtype=np.uint32)
    offsets[0::2] = np.arange(ring_size)
    offsets[1::2] = np.arange(ring_size, ring_size*2)
    offsets += 1

    for x in range(X_SAMPLES-1):
        groups.append(MeshGroup(
            PrimitiveGroupType.TRI_STRIP,
            0,
            list(offsets + x*ring_size)
        ))

    # x_max cap: note that the list needs to be reversed
    groups.append(MeshGroup(
        PrimitiveGroupType.TRI_FAN,
        0,
        [num_vertices-1]+list(range(num_vertices-2, num_vertices-ring_size-2, -1))
    ))

    return groups


MESH_GROUPS = _build_groups()


class Geometry:
    """Geometry of the Roche lobe."""

    def __init__(self, q: float, f: float):
        if f <= 0.0001:
            f = 0.0001
        vertices, radius = Geometry._compute_vertices(q, f)
        self.vertices = vertices
        self.radius = radius

    @staticmethod
    def _fix_auto_center(vertices: List[Vertex], q: float, x_min: float, x_max: float) -> float:
        # point to fix auto-centre in Celestia
        com = q/(q+1)
        com_min = com - x_min
        com_max = x_max - com
        if com_min > com_max:
            vertices.append(Vertex(
                np.array([com+com_min, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1])
            ))
            return com_min
        else:
            vertices.append(Vertex(
                np.array([com-com_max, 0, 0]), np.array([-1, 0, 0]), np.array([0.5, 0])
            ))
            return com_max

    @staticmethod
    def _compute_bounds(q: float, f: float):
        l1, l2, l3 = lagrange_points(q)
        # get dimensionless potential of common envelope
        phi1 = roche([l1, 0, 0], q)
        phi2 = roche([l2, 0, 0], q)
        phi = f*(phi2-phi1) + phi1

        # use sphere touching L2 and L3 as initial guess for outer boundary
        bound_cx = (l2+l3) / 2
        bound_r = (l2-l3) / 2

        # model in cylindrical coordinates along x-axis: get min and max extents
        x_min = _extent_min(phi, l3, q)
        x_max = _extent_max(phi, l2, q)

        # distribute points as if the envelope were two spheres touching at L1
        r1 = l1 - x_min
        r2 = x_max - l1
        minor_points = int((X_SAMPLES-1) * (2+r2/r1)/6)
        major_points = X_SAMPLES - 1 - minor_points
        x_values = np.empty(X_SAMPLES)
        # samples for primary
        x_values[:major_points] = (
            (1-np.cos(np.linspace(0, np.pi, major_points+1, endpoint=False)[1:]))*r1/2 + x_min
        )
        # sample neck
        x_values[major_points] = l1
        # samples for secondary
        x_values[major_points+1:] = (
            (1-np.cos(np.linspace(0, np.pi, minor_points+1, endpoint=False)[1:]))*r2/2 + l1
        )

        return x_min, x_max, x_values, phi

    @staticmethod
    def _compute_vertices(q: float, f: float) -> List[Vertex]:
        x_min, x_max, x_values, phi = Geometry._compute_bounds(q, f)

        # boundary for intermediate point approximation
        bound_cx = (x_min+x_max) / 2
        bound_r = (x_max-x_min) / 2

        # x_min endpoint
        vertices = [Vertex(
            np.array([x_min, 0, 0]), np.array([-1, 0, 0]), np.array([0.5, 0])
        )]

        quad_positions = np.empty((THETA_SAMPLES, 3))
        quad_normals = np.empty((THETA_SAMPLES, 3))

        # solve for intermediate points
        for x in x_values:
            v = (x-x_min)/(x_max-x_min)
            outer = np.sqrt(bound_r*bound_r - (x - bound_cx)**2)
            # compute first quadrant, saving data to mirror for other quadrants
            for theta_i in range(THETA_SAMPLES):
                rho = _surface(phi, x, theta_i, outer, q)
                position = np.array([x, rho*COS_VALS[theta_i], rho*SIN_VALS[theta_i]])
                normal = approx_fprime(position, roche, 1e-6, q)
                # force alignment of normal components at 0 and 90 degrees
                if theta_i == 0:
                    normal[2] = 0
                elif theta_i == THETA_SAMPLES-1:
                    normal[1] = 0
                # dimensionless potential increases towards the surface, so reverse for normal
                normal = -normal / norm(normal)
                quad_positions[theta_i] = position
                quad_normals[theta_i] = normal
                vertices.append(Vertex(position, normal, np.array([U_VALS[theta_i], v])))
            # second quadrant
            for theta_i in range(THETA_SAMPLES-2, 0, -1):
                vertices.append(Vertex(
                    np.multiply(quad_positions[theta_i], [1, -1, 1]),
                    np.multiply(quad_normals[theta_i], [1, -1, 1]),
                    np.array([0.5-U_VALS[theta_i], v])
                ))
            # third quadrant
            for theta_i in range(THETA_SAMPLES):
                vertices.append(Vertex(
                    np.multiply(quad_positions[theta_i], [1, -1, -1]),
                    np.multiply(quad_normals[theta_i], [1, -1, -1]),
                    np.array([0.5+U_VALS[theta_i], v])
                ))
            # fourth quadrant
            for theta_i in range(THETA_SAMPLES-2, -1, -1):
                vertices.append(Vertex(
                    np.multiply(quad_positions[theta_i], [1, 1, -1]),
                    np.multiply(quad_normals[theta_i], [1, 1, -1]),
                    np.array([1-U_VALS[theta_i], v])
                ))

        # x_max endpoint
        vertices.append(Vertex(
            np.array([x_max, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1])
        ))

        radius = Geometry._fix_auto_center(vertices, q, x_min, x_max)

        return vertices, radius


# based on modelfile.cpp
class CmodWriter:
    """Routines for writing the CMOD file."""

    def __init__(self, f: BinaryIO):
        self.f = f

    def _write_token(self, t: ModelFileToken):
        self.f.write(S_INT16.pack(t.value))

    def _write_color(self, r: float, g: float, b: float):
        self.f.write(S_COLOR.pack(ModelFileType.COLOR.value, r, g, b))

    def _write_float1(self, value: float):
        self.f.write(S_FLOAT1.pack(ModelFileType.FLOAT1.value, value))

    def _write_texture(self, source: str):
        self.f.write(S_TEXTURE.pack(
            ModelFileToken.TEXTURE.value,
            0,
            ModelFileType.STRING.value,
            len(source),
        ))
        self.f.write(source.encode('utf-8'))

    def _write_material(self, texture: str):
        self._write_token(ModelFileToken.MATERIAL)
        self._write_token(ModelFileToken.DIFFUSE)
        self._write_color(0, 0, 0)
        self._write_token(ModelFileToken.EMISSIVE)
        self._write_color(1, 1, 1)
        self._write_token(ModelFileToken.OPACITY)
        self._write_float1(1)
        self._write_texture(texture)
        self._write_token(ModelFileToken.END_MATERIAL)

    def _write_vertex_desc(
        self, semantic: VertexAttributeSemantic, format: VertexAttributeFormat
    ):
        self.f.write(S_VERTEX_DESC.pack(semantic.value, format.value))

    def _write_vertex(self, vertex: Vertex):
        # note the Geometry class uses +z as the rotation, Celestia uses +y
        self.f.write(S_VERTEX.pack(
            vertex.position[0], vertex.position[2], vertex.position[1],
            vertex.normal[0], vertex.normal[2], vertex.normal[1],
            vertex.uv[0], vertex.uv[1]
        ))

    def _write_group(self, group: MeshGroup):
        self.f.write(S_GROUP.pack(group.primitive.value, group.material, len(group.indices)))
        for index in group.indices:
            self.f.write(S_UINT32.pack(index))

    def _write_mesh(self, geometry: Geometry):
        self._write_token(ModelFileToken.MESH)

        self._write_token(ModelFileToken.VERTEX_DESC)
        self._write_vertex_desc(VertexAttributeSemantic.POSITION, VertexAttributeFormat.FLOAT3)
        self._write_vertex_desc(VertexAttributeSemantic.NORMAL, VertexAttributeFormat.FLOAT3)
        self._write_vertex_desc(VertexAttributeSemantic.TEXTURE0, VertexAttributeFormat.FLOAT2)
        self._write_token(ModelFileToken.END_VERTEX_DESC)

        self._write_token(ModelFileToken.VERTICES)
        self.f.write(S_UINT32.pack(len(geometry.vertices)))
        for vertex in geometry.vertices:
            self._write_vertex(vertex)
        for group in MESH_GROUPS:
            self._write_group(group)

        self._write_token(ModelFileToken.END_MESH)

    def write(self, geometry: Geometry, texture: str):
        """Write the given model geometry to the file."""
        self.f.write(CEL_MODEL_HEADER_BINARY)
        self._write_material(texture)
        self._write_mesh(geometry)
