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

"""Transforms to and from the ecliptic frame."""

import warnings
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

OBLIQUITY = 23.4392911
EQUATORIAL_TRANSFORM = Rotation.from_euler('x', OBLIQUITY, degrees=True)

def convert_orientation(ra: float, dec: float, inc: float) -> Tuple[float, float]:
    """Converts the inclination into Celestia's ecliptic frame."""
    # transformation between ecliptic and sky plane
    convert = Rotation.from_euler('yzx', [-dec-90, ra, -OBLIQUITY], degrees=True)
    elements = convert.inv().as_euler('zxz')  # [arg_peri, inc, node]
    elements[1] = -np.radians(inc)
    sky = Rotation.from_euler('zxz', elements)
    with warnings.catch_warnings(record=True) as wlist:
        arg_peri, inc, node = (convert*sky).as_euler('zxz', degrees=True)
        if wlist:
            # gimbal lock detected, transfer arg_peri to node (negate if retrograde)
            if inc < 90:
                node = arg_peri
            else:
                node = -arg_peri
    if node < 0:
        node += 360
    return inc, node
