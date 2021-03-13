#!/usr/bin/env python

"""Creates the W UMa catalog."""

import argparse

import astropy.io.ascii as io_ascii
from astropy.table import Table, join

from wuma_download import CATALOG_PATH, XREF_PATH, download_xref, map_names


def merge_data() -> Table:
    """Merges the W UMa and cross-reference data."""

    wuma = io_ascii.read(CATALOG_PATH, format='csv')
    map_names(wuma)
    xref = io_ascii.read(XREF_PATH, format='ecsv')
    return join(wuma, xref, keys='Name', join_type='left')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build W UMa catalog.')
    parser.add_argument('-c', '--celestia-dir', required=True, type=str)
    args = parser.parse_args()

    download_xref(args.celestia_dir)
    tbl = merge_data()
