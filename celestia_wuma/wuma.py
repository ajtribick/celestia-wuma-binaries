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

"""Creates the W UMa catalog."""

from __future__ import annotations

import os
import os.path
import string
import zipfile
from datetime import date
from typing import Optional, TextIO

import astropy.io.ascii as io_ascii
import numpy as np
from astropy.table import Row, Table, join
from astropy import units

from .download import CATALOG_PATH, XREF_PATH, download_xref, map_names
from .frame import convert_orientation
from .model import make_geometry, write_cmod
from .spparse import parse_spectrum, unparse_spectrum

VERSION = 1, 0, 2

HEADER = """# W Ursae Majoris binaries for Celestia
# -------------------------------------
#
# Generated from the WUMaCat database (Latković et al., 2021) together with
# stellar data and cross-reference information from SIMBAD and from Gaia EDR2
# (Gaia Collaboration et al., 2018). Processed for Celestia by Andrew Tribick.
#
# For further information and generation scripts, see the source repository at
# https://github.com/ajtribick/celestia-wuma-binaries
#
# References:
#
# * Astropy Collaboration et al. (2013), A&A 558, id.A33 "Astropy: A community
#   Python package for astronomy"
#
# * Bailer-Jones et al. (2018), AJ 156(2), id.58 "Estimating distances from
#   parallaxes. IV. Distances to 1.33 billion stars in *Gaia* data release 2"
#
# * Evans et al. (2018), A&A 616, id. A4 "*Gaia* Data Release 2: Photometric
#   content and validation"
#
# * Gaia Collaboration et al. (2016), A&A 595, id.A1, "The *Gaia* mission"
#
# * Gaia Collaboration et al. (2018), A&A 616, id.A1, "*Gaia* Data Release 2.
#   Summary of the contents and survey properties"
#
# * Harris et al. (2020), Nature 585, 357–362 "Array programming with NumPy"
#
# * Latković et al. (2021), "Statistics of 700 individually studied W UMa
#   stars"
#
# * Virtanen et al. (2020), Nature Methods 17(3), 261–272 "SciPy 1.0:
#   Fundamental Algorithms for Scientific Computing in Python"
#
# Acknowledgements:
#
# This work has made use of data from the European Space Agency (ESA) mission
# Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data
# Processing and Analysis Consortium (DPAC,
# https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC
# has been provided by national institutions, in particular the institutions
# participating in the Gaia Multilateral Agreement.
#
# This work has made use of the SIMBAD database, operated at CDS, Strasbourg,
# France.
#
# This work made use of [Astropy](http://www.astropy.org), a community-
# developed core Python package for Astronomy.
"""

GREEKS = [
    ('alf', 'ALF'),
    ('bet', 'BET'),
    ('gam', 'GAM'),
    ('del', 'DEL'),
    ('eps', 'EPS'),
    ('zet', 'ZET'),
    ('eta', 'ETA'),
    ('tet', 'TET'),
    ('iot', 'IOT'),
    ('kap', 'KAP'),
    ('lam', 'LAM'),
    ('mu.', 'MU'),
    ('nu.', 'NU'),
    ('ksi', 'XI'),
    ('omi', 'OMI'),
    ('pi.', 'PI'),
    ('rho', 'RHO'),
    ('sig', 'SIG'),
    ('tau', 'TAU'),
    ('ups', 'UPS'),
    ('phi', 'PHI'),
    ('chi', 'CHI'),
    ('psi', 'PSI'),
    ('ome', 'OME'),
]


def merge_data() -> Table:
    """Merges the W UMa and cross-reference data."""

    wuma = io_ascii.read(CATALOG_PATH, format='csv')
    map_names(wuma)
    xref = io_ascii.read(XREF_PATH, format='ecsv')
    tbl = join(wuma, xref, keys='Name', join_type='left')
    tbl['cel_exists'] = tbl['cel_exists'].filled(False)
    return tbl


def find_existing_names(celestia_dir: str, tbl: Table) -> dict[int, list[str]]:
    """Loads existing names for stars from starnames.dat."""
    wuma_ids = set(tbl[np.logical_not(tbl['hip'].mask)]['hip'])
    names = {}
    with open(os.path.join(celestia_dir, 'data', 'starnames.dat'), 'r') as f:
        for line in f:
            ids = line.strip().split(':')
            hip = int(ids[0])
            if hip in wuma_ids:
                names[hip] = ids[1:]
    return names


def apply_cel_convention(name: str) -> Optional[str]:
    """Applies Celestia naming conventions to a name."""
    name = ' '.join(name.split(' '))
    if name.startswith('HIP ') or name.startswith('TYC '):
        return None
    if name.startswith('BD '):
        return 'BD' + name[3:]
    if name.startswith('* '):
        name = name[2:]
    elif name.startswith('V* MU') or name.startswith('V* NU'):
        # cannot use this name in Celestia, would be interpreted as Bayer
        return None
    elif name.startswith('V* '):
        name = name[3:]
    elif name.startswith('Cl* '):
        name = name[4:]
    for greek, cel_greek in GREEKS:
        if name.startswith(greek) and len(name) > 4:
            if name[3] == '0':
                return cel_greek + name[4:]
            if name[3] == ' ' or name[3] in string.digits:
                return cel_greek + name[3:]
    return name


def _process_char(ch: str) -> str:
    if ch in string.ascii_letters or ch in string.digits:
        return ch.lower()
    if ch == '+':
        return 'p'
    if ch == '-':
        return 'm'
    if ch == '.':
        return ''
    return '_'


def _model_filename(name: str) -> str:
    return ''.join(_process_char(c) for c in name) + '.cmod'


def _format(n: float, p: int) -> str:
    # formats a float with a maximum number of decimal places
    return f'{n:.{p}f}'.rstrip('0').rstrip('.')


def _guess_spectrum(temp: float) -> str:
    # guesses spectrum from temperature, using midpoints of values in star.cpp
    if temp >= 31500:
        sptype = "O"
    elif temp >= 10010:
        sptype = "B"
    elif temp >= 7295:
        sptype = "A"
    elif temp >= 6070:
        sptype = "F"
    elif temp >= 5330:
        sptype = "G"
    elif temp >= 3895:
        sptype = "K"
    else:
        sptype = "M"
    return sptype


def _write_star_header(
    f: TextIO, row: Row, name: Optional[str], cel_names: dict[int, list[str]]
) -> Optional[tuple[Optional[str], str]]:
    # Use the spectral types with the following preference:
    # 1. SIMBAD
    # 2. The value in stars.dat, unless unknown - no override necessary here
    # 3. guess based on temperature

    if row['sp_type'] is np.ma.masked:
        sp_type = None
    else:
        sp_type = unparse_spectrum(parse_spectrum(row['sp_type']))
        if sp_type == '?':
            sp_type = None
    sp_comment = ''

    if row['cel_exists']:
        f.write(f'Modify {row["hip"]}')
        if name is not None:
            names = cel_names.get(row['hip'], [])
            if name not in names:
                f.write(f' "{":".join(names + [name])}"')
        f.write('\n{\n')
        if sp_type is None and row['needs_spectrum']:
            sp_type = _guess_spectrum(row['T1'])
            sp_comment = ' # from primary temperature'
    else:
        if row['flux'] is np.ma.masked:
            return None
        if row['hip'] is not np.ma.masked:
            f.write(f'{row["hip"]} ')
        elif name is None:
            return None
        dist = (row["dist"]*units.pc).to(units.lyr).to_value()
        f.write(f'"{name}"\n{{\n')
        f.write(f'\tRA {row["ra"]:.9}\n')
        f.write(f'\tDec {row["dec"]:.9}\n')
        f.write(f'\tDistance {dist:.9}\n')
        f.write(f'\tAppMag {_format(row["flux"], 3)}\n')
        if sp_type is None:
            sp_type = _guess_spectrum(row["T1"])
            sp_comment = ' # from primary temperature'

    return sp_type, sp_comment


def _write_common_details(
    f: TextIO, row: Row, name: Optional[str], sp_type: Optional[str], sp_comment: str
) -> None:
    if sp_type is not None:
        f.write(f'\tSpectralType "{sp_type}"{sp_comment}\n')

    f.write(f'\tTemperature {row["T1"]} # secondary = {row["T2"]} K\n')

    geometry = make_geometry(row['q'], row['f'])

    f.write(f'\tRadius {row["a"] * geometry.radius * 696000:.0f}\n')
    meshname = _model_filename(name if name is not None else row['Name'])
    f.write(f'\tMesh "{meshname}"\n')
    f.write('\tUniformRotation {\n')
    f.write(f'\t\tPeriod {row["P"]*24:.9}\n')

    inc, node = convert_orientation(row['ra'], row['dec'], row['i'])
    f.write(f'\t\tInclination {inc:.3f}\n')
    f.write(f'\t\tAscendingNode {node:.3f}\n')

    f.write('\t}\n}\n')

    with open(os.path.join('output', 'models', meshname), 'wb') as mf:
        write_cmod(mf, geometry)


def create_stars(celestia_dir: str, f: TextIO, tbl: Table) -> None:
    """Creates the star data."""
    print("Writing output files")
    tbl = tbl[np.logical_not(np.logical_or(tbl['a'].mask, tbl['dist'].mask))]
    cel_names = find_existing_names(celestia_dir, tbl)
    total_output = 0
    for row in tbl:
        f.write(
            f'\n# {row["Name"]}: q={_format(row["q"], 3)}, a={_format(row["a"], 2)}, '
            f'f={_format(row["f"], 6)}\n'
        )
        name = apply_cel_convention(row['Name'])

        spectrum_info = _write_star_header(f, row, name, cel_names)
        if spectrum_info is None:
            continue

        sp_type, sp_comment = spectrum_info
        _write_common_details(f, row, name, sp_type, sp_comment)
        total_output += 1

    print(f'Output {total_output} binaries')


def build_catalog(celestia_dir: str) -> None:
    """Builds the W UMa catalog."""
    download_xref(celestia_dir)

    try:
        os.makedirs('output/models')
    except FileExistsError:
        pass

    tbl = merge_data()
    with open(os.path.join('output', 'wuma.stc'), 'w') as f:
        f.write(HEADER)
        f.write(f'\n# Generated on {date.today().isoformat()}\n')
        create_stars(celestia_dir, f, tbl)

    print("Creating archive")
    archive_name = f'wuma-{VERSION[0]}.{VERSION[1]}.{VERSION[2]}'
    with zipfile.ZipFile(
        f'{archive_name}.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        for root, _dirs, files in os.walk('output'):
            for file in files:
                fs_path = os.path.join(root, file)
                z.write(fs_path, os.path.join(archive_name, os.path.relpath(fs_path, 'output')))
