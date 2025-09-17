# Refactored plume-vetting capability:

Notable differences from initial iteration include:

- Adherence to Python packaging best practices (https://packaging.python.org/en/latest/). Plume vetting code now installs as a package with all dependencies, e.g., "pip install ." (venvs recommended).
- All code configuration settings gathered in ./config/config.yaml.
- Simplified filestring, glob, and data/metadata access via EMIT acquisition file (./src/pv/emit_file.EMITAcquisitionFile) and matched filter file classes (./src/pv/emit_file.EMITMatchedFilterFile).
- Simplified/non-redundant plume ingest and data handling via GeoJSON and GeoPandas objects, encapsulated in EMITPlume class, along with common plume vetting operations as class-based functionality (e.g., affine transformations, plume boundary variation, plume mask determination, etc.).
- Georectified lon/lat and x/y pixel affine transformations using L1B GLT header file coefficients, thus eliminating calls to gdal.GetGeoTransform.
- Single top-level application to generate plume vetting metrics for selected plumes (in progress).
- All pixel-level operations now consistently handled as mask-based operations.
- Target pixel "dilation" (part of spectral pairing) now as Gaussian filter image convolution.
