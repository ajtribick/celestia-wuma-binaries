"""Transforms to and from the ecliptic frame."""

import numpy as np
from scipy.spatial.transform import Rotation

OBLIQUITY = 23.4392911
EQUATORIAL_TRANSFORM = Rotation.from_euler('x', OBLIQUITY, degrees=True)

def convert_orientation(ra: float, dec: float, inc: float) -> np.ndarray:
    """Converts the inclination into Celestia's ecliptic frame."""
    # transformation between ecliptic and sky plane
    convert = Rotation.from_euler('yzx', [-dec-90, ra, -OBLIQUITY], degrees=True)
    elements = convert.inv().as_euler('zxz')  # [arg_peri, inc, node]
    elements[1] = -np.radians(inc)
    sky = Rotation.from_euler('zxz', elements)
    return (convert*sky).as_euler('zxz', degrees=True)  # [arg_peri, inc, node]
