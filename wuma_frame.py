"""Transforms to and from the ecliptic frame."""

from scipy.spatial.transform import Rotation

OBLIQUITY = 23.4392911
EQUATORIAL_TRANSFORM = Rotation.from_euler('x', OBLIQUITY, degrees=True)
ECLIPTIC_TRANSFORM = Rotation.from_euler('x', -OBLIQUITY, degrees=True)
