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

"""Downloads the data files for the W UMa catalog."""

import gzip
import os
import os.path
import struct
from typing import BinaryIO, Mapping, Optional, Tuple

import astropy.io.ascii as io_ascii
import numpy as np
import requests
from astropy.io import votable
from astropy.table import Column, MaskedColumn, Table, join, vstack
from astroquery.gaia import Gaia
from pyvo.dal import AsyncTAPJob
from scipy.linalg import norm

from .frame import EQUATORIAL_TRANSFORM
from .spparse import CelMkClass

CATALOG_URL = 'https://wumacat.aob.rs/Downloads/Catalog'
SIMBAD_TAP_URL = 'http://simbad.u-strasbg.fr:80/simbad/sim-tap'

CATALOG_PATH = os.path.join('data', 'WUMaCat.csv')
SIMBAD_PATH = os.path.join('data', 'simbad.votable')
GAIA_PATH = os.path.join('data', 'gaia.votable.gz')
XREF_PATH = os.path.join('data', 'xref.ecsv')

MAP_NAMES = [
    ('44 Boo', '44 Boo B'),
]

def download_catalog() -> None:
    """Downloads the W UMa catalog"""

    if os.path.isfile(CATALOG_PATH):
        print('W UMa catalog already downloaded, skipping')
        return

    print(f'Downloading catalog from {CATALOG_URL}')
    with requests.get(CATALOG_URL, stream=True) as r, \
         open(CATALOG_PATH, 'wb') as f:
        for chunk in r.iter_content():
            f.write(chunk)


def map_names(tbl: Table) -> None:
    """Applies name fixes."""

    names = tbl['Name']
    for old_name, new_name in MAP_NAMES:
        names[names == old_name] = new_name


def download_simbad() -> None:
    """Downloads the SIMBAD data"""
    if os.path.isfile(SIMBAD_PATH):
        print('SIMBAD result already downloaded, skipping')
        return

    print('Querying SIMBAD')

    names = io_ascii.read(CATALOG_PATH, format='csv', include_names=['Name'])
    map_names(names)
    query = r"""
        SELECT
            w.Name, b.main_id, b.ra, b.dec, f.flux, b.sp_type, ii.ids
        FROM
            TAP_UPLOAD.WUMA_NAMES w
            JOIN IDENT i ON i.id = w.Name
            JOIN BASIC b ON b.oid = i.oidref
            JOIN IDS ii ON ii.oidref = b.oid
            LEFT JOIN FLUX f ON f.oidref = b.oid AND f.filter = 'V'
    """

    job = AsyncTAPJob.create(SIMBAD_TAP_URL, query, uploads={'WUMA_NAMES': names})
    job.run().wait()
    job.raise_if_error()
    with requests.get(job.result_uri, stream=True) as r, \
         open(SIMBAD_PATH, 'wb') as f:
        for chunk in r.iter_content():
            f.write(chunk)
    job.delete()


def _map_ids(tbl: Table) -> Mapping[int, int]:
    gaia_ids = np.zeros(len(tbl), dtype=np.int64)
    cel_ids = np.zeros(len(tbl), dtype=np.uint32)
    id_idx = {}

    for i, ids in enumerate(tbl['ids']):
        for ident in ids.split('|'):
            if ident.startswith('HIP'):
                hip = int(ident[3:].strip())
                assert hip not in id_idx
                id_idx[hip] = i
                cel_ids[i] = hip
            elif ident.startswith('TYC'):
                tycs = [int(t) for t in ident[3:].strip().split('-')]
                tyc = tycs[0] + tycs[1]*10000 + tycs[2]*1000000000
                assert tyc not in id_idx
                id_idx[tyc] = i
                if cel_ids[i] == 0:
                    cel_ids[i] = tyc
            elif ident.startswith('Gaia DR2'):
                gaia_ids[i] = int(ident[8:].strip())

    tbl.add_columns([
        MaskedColumn(data=gaia_ids, name='gaia'),
        MaskedColumn(data=cel_ids, name='hip'),
    ])

    return id_idx


def _check_header(f: BinaryIO) -> int:
    header = f.read(14)
    if len(header) != 14:
        raise EOFError("Unexpected end-of-file")
    db_type, db_version, db_len = struct.unpack('<8sHI', header)
    if db_type != b'CELSTARS' or db_version != 0x0100:
        raise ValueError("Bad header format")
    return db_len


_STAR_FORMAT = struct.Struct('<I3fhH')


def _process_star(
    f: BinaryIO,
    id_idx: Mapping[int, int],
) -> Optional[Tuple[int, int, float, float, float, int]]:
    star = f.read(20)
    if len(star) != 20:
        raise EOFError("Unexpected end-of-file")
    hip, x, y, z, _v_mag, sp_type = _STAR_FORMAT.unpack(star)
    try:
        idx = id_idx[hip]
    except KeyError:
        return None

    pos = EQUATORIAL_TRANSFORM.apply([x, y, z])
    dist = norm(pos)
    ra = np.degrees(np.arctan2(-pos[2], pos[0]))
    if ra < 0:
        ra += 2*np.pi
    dec = np.degrees(np.arcsin(pos[1] / dist))

    return idx, hip, ra, dec, dist, sp_type


def apply_celestia(celestia_dir: str) -> Table:
    """Adds RA and Dec information from Celestia."""

    print("Cross-matching to stars.dat")
    tbl = votable.parse_single_table(SIMBAD_PATH).to_table()

    tbl['Name'] = tbl['Name'].astype('U')
    tbl['main_id'] = tbl['main_id'].astype('U')
    tbl['sp_type'] = tbl['sp_type'].astype('U')
    tbl['sp_type'].mask = np.logical_or(tbl['sp_type'].mask, tbl['sp_type'] == '')

    id_idx = _map_ids(tbl)

    tbl.add_columns([
        MaskedColumn(data=np.full(len(tbl), np.nan, dtype=np.float64), name='dist'),
        Column(data=np.full(len(tbl), False, dtype=np.bool_), name='cel_exists'),
        Column(data=np.full(len(tbl), False, dtype=np.bool_), name='needs_spectrum'),
    ])

    with open(os.path.join(celestia_dir, 'data', 'stars.dat'), 'rb') as f:
        db_len = _check_header(f)
        while db_len > 0:
            db_len -= 1
            star = _process_star(f, id_idx)
            if star is None:
                continue
            idx, hip, ra, dec, dist, sp_type = star

            tbl['ra'][idx] = ra
            tbl['dec'][idx] = dec
            tbl['dist'][idx] = dist

            if not tbl['cel_exists'][idx]:
                tbl['cel_exists'][idx] = True
                tbl['hip'][idx] = hip
                if (sp_type & 0xff00) == CelMkClass.UNKNOWN:
                    tbl['needs_spectrum'][idx] = True

    tbl['dist'].mask = np.isnan(tbl['dist'])
    tbl['hip'].mask = tbl['hip'] == 0
    tbl['gaia'].mask = tbl['gaia'] == 0

    return tbl


def download_gaia(tbl: Table) -> None:
    """Downloads Gaia data for stars not in catalog."""

    if os.path.isfile(GAIA_PATH):
        print('Gaia data already downloaded, skipping')
        return

    print("Querying Gaia")

    source_ids = Table([
        tbl[np.logical_not(np.logical_or(tbl['cel_exists'], tbl['gaia'].mask))]['gaia']
    ])
    query = r"""
        SELECT
            g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp, d.r_est
        FROM
            TAP_UPLOAD.source_ids s
            JOIN gaiadr2.gaia_source g ON g.source_id = s.gaia
            LEFT JOIN external.gaiadr2_geometric_distance d ON d.source_id = g.source_id
    """

    job = Gaia.launch_job_async(
        query, output_file=GAIA_PATH, output_format='votable', dump_to_file=True,
        upload_resource=source_ids, upload_table_name='source_ids'
    )
    job.wait_for_job_end()


def merge_gaia(tbl: Table) -> Table:
    """Merges Gaia data for non-Celestia stars."""
    with gzip.open(GAIA_PATH, 'rb') as f:
        gaia = votable.parse_single_table(f).to_table()

    bp_rp = gaia['bp_rp'].filled(0)
    bp_rp2 = bp_rp*bp_rp

    gaia.add_column(MaskedColumn(
        data=gaia['phot_g_mean_mag'].filled(np.nan) + 0.01760 + bp_rp*0.006860 + bp_rp2*0.1732,
        name='flux',
        mask=gaia['phot_g_mean_mag'].mask
    ))

    gaia.remove_columns(['phot_g_mean_mag', 'bp_rp'])
    gaia.rename_column('source_id', 'gaia')
    gaia.rename_column('r_est', 'dist')

    has_gaia = tbl[np.logical_not(tbl['gaia'].mask)]
    merged = join(has_gaia, gaia, keys=['gaia'], join_type='left', table_names=['cel', 'gaia'])
    merged['ra'] = np.where(merged['ra_gaia'].mask, merged['ra_cel'], merged['ra_gaia'])
    merged['dec'] = np.where(merged['dec_gaia'].mask, merged['dec_cel'], merged['dec_gaia'])
    merged.add_columns([
        MaskedColumn(
            data=np.where(merged['dist_gaia'].mask, merged['dist_cel'], merged['dist_gaia']),
            name='dist',
            mask=np.logical_and(merged['dist_gaia'].mask, merged['dist_cel'].mask)
        ),
        MaskedColumn(
            data=np.where(merged['flux_cel'].mask, merged['flux_gaia'], merged['flux_cel']),
            name='flux',
            mask=np.logical_and(merged['flux_cel'].mask, merged['flux_gaia'].mask)
        )
    ])
    merged.remove_columns([
        'ra_cel', 'ra_gaia', 'dec_cel', 'dec_gaia', 'dist_cel', 'dist_gaia', 'flux_cel', 'flux_gaia'
    ])

    return vstack([tbl[tbl['gaia'].mask], merged], join_type='exact')


def download_xref(celestia_dir: str) -> None:
    """Downloads cross-reference data."""

    try:
        os.mkdir('data')
    except FileExistsError:
        pass

    if os.path.isfile(XREF_PATH):
        print('Cross-reference data already downloaded, skipping')
        return

    download_catalog()
    download_simbad()
    tbl = apply_celestia(celestia_dir)
    download_gaia(tbl)
    tbl = merge_gaia(tbl)
    tbl.write(XREF_PATH, format='ascii.ecsv')
