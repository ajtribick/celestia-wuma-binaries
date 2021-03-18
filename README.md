# W Ursae Majoris binaries for Celestia

This repository contains scripts to generate W Ursae Majoris binaries for
Celestia, based on the [WUMaCat](https://wumacat.aob.rs) database. Each binary
is represented as a single star in Celestia, with a custom mesh giving it the
shape of the common envelope.

Spectral types, apparent magnitudes and cross-matching of identifiers to the
Hipparcos, Tycho and Gaia DR2 catalogues are obtained from SIMBAD. Where
V-magnitudes are not available, this is estimated using the relationships given
in Evans et al. (2018). Distances for stars not included in Celestia's
stars.dat file are taken from the Gaia DR2 geometric distance catalogue
(Bailer-Jones et al., 2018).

Pre-built releases for given versions of the
[CelestiaContent](https://github.com/CelestiaProject/CelestiaContent)
repository are available. The below instructions are provided for re-generating
the files in case of updates to Celestia.

## Prerequisites

- Celestia
- Python 3.8 or higher
- Internet connection

## How to use

1. Clone or download this repository.
2. Open a command window in the repository directory.
3. Set up a Python 3 virtual environment

   `python3 -m venv myenv`

4. Switch to the virtual environment.

   `source myenv/bin/activate` (Linux)
   `.\myenv\Scripts\Activate.ps1` (Windows Powershell)

5. Install the requirements.

   `python -m pip install -r requirements.txt`

6. Run the script, supplying the path to the Celestia directory (this is used
   for star names and positions).

   `python wuma.py -c (path to Celestia)`

7. The output is generated in the archive `wuma.zip`.

## Generating custom contact binary meshes

Custom contact binary meshes can be generated using the classes in the
`wuma_model.py`. For example:

```python
from wuma_model import CmodWriter, Geometry

q = 0.3  # mass ratio
f = 0.6  # fillout factor

g = Geometry(q, f)
print(f'Radius factor {g.radius}')

with open('output.cmod', 'wb') as f:
   c = CmodWriter(f)
   c.write(g)
```

The "radius factor" is a value which should be multiplied by the binary's
semimajor axis and used as the `Radius` parameter in the .stc file.

In order to produce an orientation with a given inclination, use the
`convert_orientation` function in `wuma_frame.py`:

```python
from wuma_frame import convert_orientation

ra = 123.45  # in degrees
dec = -67.8  # in degrees
inc = 90.12  # in degrees

inc_cel, node_cel = convert_orientation(ra, dec, inc)
```

The values in `inc_cel` and `node_cel` can be used as the `Inclination` and
`AscendingNode` parameters in the `UniformRotation` block in the .stc file.

## References

* Astropy Collaboration et al. (2013), A&A 558, id.A33 "Astropy: A community
  Python package for astronomy"
* Bailer-Jones et al. (2018), AJ 156(2), id.58 "Estimating distances from
  parallaxes. IV. Distances to 1.33 billion stars in *Gaia* data release 2"
* Evans et al. (2018), A&A 616, id. A4 "*Gaia* Data Release 2: Photometric
  content and validation"
* Gaia Collaboration et al. (2016), A&A 595, id.A1, "The *Gaia* mission"
* Gaia Collaboration et al. (2018), A&A 616, id.A1, "*Gaia* Data Release 2.
  Summary of the contents and survey properties"
* Harris et al. (2020), Nature 585, 357–362 "Array programming with NumPy"
* Latković et al. (2021), "Statistics of 700 individually studied W UMa stars"
* Virtanen et al. (2020), Nature Methods 17(3), 261–272 "SciPy 1.0: Fundamental
  Algorithms for Scientific Computing in Python"

## Acknowledgements

This work has made use of data from the European Space Agency (ESA) mission
[*Gaia*](https://www.cosmos.esa.int/gaia), processed by the *Gaia* Data
Processing and Analysis Consortium
([DPAC](https://www.cosmos.esa.int/web/gaia/dpac/consortium)). Funding for the
DPAC has been provided by national institutions, in particular the institutions
participating in the *Gaia* Multilateral Agreement.

This work has made use of the SIMBAD database, operated at CDS, Strasbourg,
France.

This work made use of [Astropy](http://www.astropy.org), a community-developed
core Python package for Astronomy.
