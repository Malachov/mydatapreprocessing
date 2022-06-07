# List of what have been done in new versions

## v3.0
- Refactored (backward incompatible). Divided into several subpackages. Mainly new 'consolidation'. There are new main pipelining functions and functions are implemented aside (also can be used separately). 
- All functions moved to consolidation subpackage now supports inplace parameter which may result in better performance. 

## v2.0
- Refactored (backward incompatible) - now data loading functions are in `load_data` module and feature engineering functions are in `feature_engineering` module. Various functions not related to used modules are in `misc`.
- Database module updated and tests done
- Clean DataFrame to table print function
- Type hints

## v1.1

- New formats support. H5, Parquet, xlsx...
- Multiple files at once in a list ['file1.csv', 'file2.csv']
- New function get_file_paths that open dialog window and let you choose files very simply. Return list of paths.
- String embedding - transform string columns - E.g. Country ('US', 'FR'... etc.) into numbers so it can be used in machine learning models.
- Deriving new columns with rolling fast fourier transform in preprocessing with add_frequency_columns()
- Deriving new columns with add_derived_columns() in preprocessing such as first and second difference or rolling means and rolling standard deviations
- Inputs module described in readme
- No predicted column have to be specified in consolidation
- Binning preprocessing function added

## v1.0

- Realease version, CI and docs deployment
