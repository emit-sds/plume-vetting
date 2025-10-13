# Refactored plume-vetting test cases and examples

Test cases that can be run with pytest, etc. are prefixed with
'test\_', while demonstration examples are prefixed with 'demo\_'.

## Contents:

### Scripts, data:

- config.yaml: Example configuration file, with appropriate test case
and demonstration example settings.

- get\_test\_data.sh: Helper script for downloading data used in test
cases and examples. See script for further information, and note that
values for "username" and "remote" need to be provided.

### Test cases:

- test\_emit\_plume.py: Test various methods of initializing EMITPlume
class.

- test\_transmittance\_model.py: Simple transmittance model test with
some made-up numbers.

### Demonstration examples:

- demo\_plume\_latlon\_to\_pixels.py: Test conversion of lat/lon plume
boundary to pixel coordinates, and visually confirm via retrieval and
orthorectified image plots.

- demo\_plume\_pixel\_pairing.ipynb: Walks through entire process of
optimal target and background pixel pairing.

- demo\_plume\_variations.py: Test generation of random plume
variations and, optionally, show plume mask components.

