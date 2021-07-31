# List of what could be done

- [ ] Add features
    use_last_line (bool, optional): Last value may have not been complete yet. If False, it's removed. Defaults to False.
    reversit (bool, optional): If want to limit number of loaded lines, you can use select top and use reversed order.
        If you need to reverse it back, set this to True. Defaults to False
- [ ] Add Sqlite and postgres to databases 
- [ ] Join load and consolidate to get_data
- [ ] Rename to mydataprocessing - load data and data consolidation to own submodule - get_data
- [ ] Defince inplace argument in consolidation, data_preprocessing, and do inplace operations where possible
- [ ] In data consolidation, return also full scaler for all columns for inverse all columns and return in class maybe
- [ ] Remove input_type_name and replace with one_column option
- [ ] Feature extraction
- [ ] Finish generate test data modul -rename to datasets - add some real data, ramp etc...
- [ ] Add unit test for inputs (from visual) and generation
- [ ] Add examples to docstrings
- [ ] In remove_outliers function do not remove it, but interpolate by neighbors option 