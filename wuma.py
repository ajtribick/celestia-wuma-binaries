#!/usr/bin/env python

"""Creates the W UMa catalog."""

import argparse
import os
import os.path
import string
import sys
from typing import Dict, List, Optional, TextIO

import astropy.io.ascii as io_ascii
from astropy.table import Table, join
from astropy import units

import numpy as np

from spparse import parse_spectrum, unparse_spectrum
from wuma_download import CATALOG_PATH, XREF_PATH, download_xref, map_names
from wuma_model import CmodWriter, Geometry


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


def find_existing_names(celestia_dir: str, tbl: Table) -> Dict[int, List[str]]:
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


def process_char(ch: str) -> str:
    if ch in string.ascii_letters or ch in string.digits:
        return ch.lower()
    if ch == '+':
        return 'p'
    if ch == '-':
        return 'm'
    if ch == '.':
        return ''
    return '_'

def model_filename(name: str) -> str:
    return ''.join(process_char(c) for c in name) + '.cmod'


def create_stars(celestia_dir: str, f: TextIO, tbl: Table):
    """Creates the star data."""
    tbl = tbl[np.logical_not(np.logical_or(tbl['a'].mask, tbl['dist'].mask))]
    cel_names = find_existing_names(celestia_dir, tbl)
    for row in tbl:
        f.write(f'\n# {row["Name"]}\n')
        name = apply_cel_convention(row['Name'])
        if row['cel_exists']:
            f.write(f'Modify {row["hip"]}')
            if name is not None:
                names = cel_names.get(row['hip'], [])
                if name not in names:
                    f.write(f' "{":".join(names + [name])}"')
            f.write('\n{\n')
        else:
            if row['hip'] is not np.ma.masked:
                f.write(f'{row["hip"]} ')
            elif name is None:
                continue
            dist = (row["dist"]*units.pc).to(units.lyr).to_value()
            f.write(f'"{name}"\n{{\n')
            f.write(f'\tRA {row["ra"]}\n')
            f.write(f'\tDec {row["dec"]}\n')
            f.write(f'\tDistance {dist}\n')
            f.write(f'\tAppMag {row["flux"]}\n')
            if row["sp_type"] is np.ma.masked:
                f.write('\tSpectralType "?"\n')
        if row["sp_type"] is not np.ma.masked:
            # Celestia's spectral type parser is relatively basic, so use the parser from the
            # Gaia DR2 add-on and then unparse it back to a Celestia type
            sp_type = unparse_spectrum(parse_spectrum(row["sp_type"]))
            f.write(f'\tSpectralType "{sp_type}"\n')
        f.write(f'\tTemperature {row["T1"]} # from primary\n')
        f.write(f'\tRadius {row["a"] * 696000}\n')
        meshname = model_filename(name if name is not None else row['Name'])
        f.write(f'\tMesh "{meshname}"\n')
        f.write(f'\tTexture "wuma.jpg"\n')
        f.write(f'\tUniformRotation {{\n')
        f.write(f'\t\tPeriod {row["P"]*24}\n')
        f.write('\t}\n}\n')

        geometry = Geometry(row['q'], row['f'])
        with open(os.path.join('output', 'models', meshname), 'wb') as mf:
            writer = CmodWriter(mf)
            writer.write(geometry, "wuma.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build W UMa catalog.')
    parser.add_argument('-c', '--celestia-dir', required=True, type=str)
    args = parser.parse_args()

    download_xref(args.celestia_dir)

    try:
        os.makedirs('output/models')
    except FileExistsError:
        pass

    tbl = merge_data()
    with open(os.path.join('output', 'wuma.stc'), 'w') as f:
        create_stars(args.celestia_dir, f, tbl)
