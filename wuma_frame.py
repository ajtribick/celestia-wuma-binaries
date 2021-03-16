"""Transforms to and from the ecliptic frame."""

from typing import Tuple
import warnings

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
