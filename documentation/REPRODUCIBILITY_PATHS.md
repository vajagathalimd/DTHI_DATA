# Reproducibility paths

Public analysis scripts do not contain machine-specific local filesystem paths.

Where a script requires the original DTHI project directory structure, set:

`DTHI_PROJECT_ROOT=/path/to/DTHI_Struct_Cortical_Hierarchy`

If this variable is not supplied, scripts using this convention default to the
current working directory.

The public repository contains derived manuscript-supporting data, audit files,
and code. Large third-party datasets are not redistributed and must be obtained
from their original repositories.
