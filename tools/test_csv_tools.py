import csv_tools
def test_iterable_to_csv_single_line_with_escape():
    import file_tools
    # @file_tools.print_to_file(PROJ_ABS_FILEPATH+"./tools/output.csv")
    def _():
        d = [{"name": "Alice", "age": 30, "city": 'Wonderland|City"'}, {"name": "Bob", "age": 25, "city": 'Bob|City"'}, {"name": "Charlie", "age": 35, "city": 'Charlie|City"'}]
        print(csv_tools.iterable_to_csv_single_line_with_escape(d[0].keys()))
        for item in d:
            print(csv_tools.iterable_to_csv_single_line_with_escape(item.values()))
    _()
    
def test_csv_to_dict_single_line_with_escape():
    content = 'name|age|city\nAlice|30|"Wonderland|City"""\nBob|25|"Bob|City"""\nCharlie|35|"Charlie|City"""'
    c = content.split('\n')
    header = csv_tools._csv_to_innerlist_single_line_with_escape(c[0])
    for i in c[1:]:
        actual = csv_tools.csv_to_dict_single_line_with_escape(header,i)
        print(actual)     