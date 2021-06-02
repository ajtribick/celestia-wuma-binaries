# celestia-wuma-binaries: W Ursae Majoris binaries for Celestia
# Copyright (C) 2019–2020  Andrew Tribick
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Writes out the cmod file."""

import struct
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import norm
from scipy.optimize import approx_fprime, brentq, newton

CEL_MODEL_HEADER_BINARY = b"#celmodel_binary"


# from modelfile.h
class _ModelFileToken(Enum):
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
class _ModelFileType(Enum):
    FLOAT1 = 1
    FLOAT2 = 2
    FLOAT3 = 3
    FLOAT4 = 4
    STRING = 5
    UINT32 = 6
    COLOR = 7


# from mesh.h
class _VertexAttributeSemantic(Enum):
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
class _VertexAttributeFormat(Enum):
    FLOAT1 = 0
    FLOAT2 = 1
    FLOAT3 = 2
    FLOAT4 = 3
    UBYTE4 = 4


# from mesh.h
class _PrimitiveGroupType(Enum):
    TRI_LIST = 0
    TRI_STRIP = 1
    TRI_FAN = 2
    LINE_LIST = 3
    LINE_STRIP = 4
    POINT_LIST = 5
    SPRITE_LIST = 6


_S_INT16 = struct.Struct('<h')
_S_UINT32 = struct.Struct('<I')
_S_COLOR = struct.Struct('<h3f')
_S_FLOAT1 = struct.Struct('<hf')
_S_TEXTURE = struct.Struct('<4h')
_S_VERTEX_DESC = struct.Struct('<2h')
_S_VERTEX = struct.Struct('<8f')
_S_GROUP = struct.Struct('<h2I')


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


@dataclass
class Vertex:
    """Vertex attributes."""
    position: np.ndarray
    normal: np.ndarray
    uv: np.ndarray


@dataclass
class _MeshGroup:
    primitive: _PrimitiveGroupType
    material: int
    indices: List[int]


# pre-build groups
def _build_groups() -> List[_MeshGroup]:
    ring_size = 4*THETA_SAMPLES - 3
    num_vertices = 2 + ring_size*X_SAMPLES

    # x_min cap
    groups = [_MeshGroup(_PrimitiveGroupType.TRI_FAN, 0, list(range(ring_size+1)))]

    # triangle strip goes ring 1 point 1, ring 2 point 1, ring 1 point 2, ring 2 point 2, etc.
    offsets = np.empty(ring_size*2, dtype=np.uint32)
    offsets[0::2] = np.arange(ring_size)
    offsets[1::2] = np.arange(ring_size, ring_size*2)
    offsets += 1

    for x in range(X_SAMPLES-1):
        groups.append(_MeshGroup(
            _PrimitiveGroupType.TRI_STRIP,
            0,
            list(offsets + x*ring_size)
        ))

    # x_max cap: note that the list needs to be reversed
    groups.append(_MeshGroup(
        _PrimitiveGroupType.TRI_FAN,
        0,
        [num_vertices-1]+list(range(num_vertices-2, num_vertices-ring_size-2, -1))
    ))

    return groups


_MESH_GROUPS = _build_groups()


@dataclass
class Geometry:
    """Geometry of the Roche lobe."""
    vertices: List[Vertex]
    radius: float


def _fix_auto_center(vertices: List[Vertex], q: float, x_min: float, x_max: float) -> float:
    # point to fix auto-centre in Celestia
    com = q/(q+1)
    com_min = com - x_min
    com_max = x_max - com
    if com_min > com_max:
        vertices.append(Vertex(
            np.array([com+com_min, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1])
        ))
        radius = com_min
    else:
        vertices.append(Vertex(
            np.array([com-com_max, 0, 0]), np.array([-1, 0, 0]), np.array([0.5, 0])
        ))
        radius = com_max

    return radius


@dataclass
class _GeometryBounds:
    x_min: float
    x_max: float
    x_values: np.ndarray
    phi: float
    cx: float
    r: float


def _compute_x_values(x_min: float, x_max: float, l1: float) -> np.ndarray:
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

    return x_values


def _compute_bounds(q: float, f: float) -> _GeometryBounds:
    l1, l2, l3 = lagrange_points(q)
    # get dimensionless potential of common envelope
    phi1 = roche([l1, 0, 0], q)
    phi2 = roche([l2, 0, 0], q)
    phi = f*(phi2-phi1) + phi1

    # model in cylindrical coordinates along x-axis: get min and max extents
    x_min = _extent_min(phi, l3, q)
    x_max = _extent_max(phi, l2, q)

    x_values = _compute_x_values(x_min, x_max, l1)

    # boundary for intermediate point approximation
    cx = (x_min+x_max) / 2
    r = (x_max-x_min) / 2

    return _GeometryBounds(x_min, x_max, x_values, phi, cx, r)


def make_geometry(q: float, f: float) -> Geometry:
    """Creates the geometry for a W UMa binary."""

    bounds = _compute_bounds(q, max(f, 0.0001))

    # x_min endpoint
    vertices = [
        Vertex(np.array([bounds.x_min, 0, 0]), np.array([-1, 0, 0]), np.array([0.5, 0])),
    ]

    quad_positions = np.empty((THETA_SAMPLES, 3))
    quad_normals = np.empty((THETA_SAMPLES, 3))

    # solve for intermediate points
    for x in bounds.x_values:
        v = (x-bounds.x_min)/(bounds.x_max-bounds.x_min)
        outer = np.sqrt(bounds.r*bounds.r - (x - bounds.cx)**2)
        # compute first quadrant, saving data to mirror for other quadrants
        for theta_i in range(THETA_SAMPLES):
            rho = _surface(bounds.phi, x, theta_i, outer, q)
            position = np.array([x, rho*COS_VALS[theta_i], rho*SIN_VALS[theta_i]])
            normal = approx_fprime(position, roche, 1e-6, q)
            # force alignment of normal components at 0 and 90 degrees
            if theta_i == 0:
                normal[2] = 0
            elif theta_i == THETA_SAMPLES-1:
                normal[1] = 0
            # dimensionless potential increases towards the surface, so reverse for normal
            normal /= -norm(normal)
            quad_positions[theta_i] = position
            quad_normals[theta_i] = normal
            vertices.append(Vertex(
                position,
                normal,
                np.array([U_VALS[theta_i], v])
            ))
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
        np.array([bounds.x_max, 0, 0]), np.array([1, 0, 0]), np.array([0.5, 1])
    ))

    radius = _fix_auto_center(vertices, q, bounds.x_min, bounds.x_max)

    return Geometry(vertices, radius)


# CMOD writing utilities, based on modelfile.cpp

def _write_token(f: BinaryIO, t: _ModelFileToken) -> None:
    f.write(_S_INT16.pack(t.value))


def _write_color(f: BinaryIO, r: float, g: float, b: float) -> None:
    f.write(_S_COLOR.pack(_ModelFileType.COLOR.value, r, g, b))


def _write_float1(f: BinaryIO, value: float) -> None:
    f.write(_S_FLOAT1.pack(_ModelFileType.FLOAT1.value, value))


def _write_material(f: BinaryIO) -> None:
    _write_token(f, _ModelFileToken.MATERIAL)
    _write_token(f, _ModelFileToken.DIFFUSE)
    _write_color(f, 0, 0, 0)
    _write_token(f, _ModelFileToken.EMISSIVE)
    _write_color(f, 1, 1, 1)
    _write_token(f, _ModelFileToken.OPACITY)
    _write_float1(f, 1)
    _write_token(f,_ModelFileToken.END_MATERIAL)


def _write_vertex_desc(
    f: BinaryIO,
    semantic: _VertexAttributeSemantic,
    fmt: _VertexAttributeFormat,
) -> None:
    f.write(_S_VERTEX_DESC.pack(semantic.value, fmt.value))


def _write_vertex(f: BinaryIO, vertex: Vertex) -> None:
    # note the Geometry class uses +z as the rotation, Celestia uses +y
    f.write(_S_VERTEX.pack(
        vertex.position[0], vertex.position[2], vertex.position[1],
        vertex.normal[0], vertex.normal[2], vertex.normal[1],
        vertex.uv[0], vertex.uv[1]
    ))


def _write_group(f: BinaryIO, group: _MeshGroup) -> None:
    f.write(_S_GROUP.pack(group.primitive.value, group.material, len(group.indices)))
    for index in group.indices:
        f.write(_S_UINT32.pack(index))


def _write_mesh(f: BinaryIO, geometry: Geometry) -> None:
    _write_token(f, _ModelFileToken.MESH)

    _write_token(f, _ModelFileToken.VERTEX_DESC)
    _write_vertex_desc(f, _VertexAttributeSemantic.POSITION, _VertexAttributeFormat.FLOAT3)
    _write_vertex_desc(f, _VertexAttributeSemantic.NORMAL, _VertexAttributeFormat.FLOAT3)
    _write_vertex_desc(f, _VertexAttributeSemantic.TEXTURE0, _VertexAttributeFormat.FLOAT2)
    _write_token(f, _ModelFileToken.END_VERTEX_DESC)

    _write_token(f, _ModelFileToken.VERTICES)
    f.write(_S_UINT32.pack(len(geometry.vertices)))
    for vertex in geometry.vertices:
        _write_vertex(f, vertex)
    for group in _MESH_GROUPS:
        _write_group(f, group)

    _write_token(f, _ModelFileToken.END_MESH)


def write_cmod(f: BinaryIO, geometry: Geometry) -> None:
    """Write the given model geometry to the file."""
    f.write(CEL_MODEL_HEADER_BINARY)
    _write_material(f)
    _write_mesh(f, geometry)
