import csv
from typing import Iterable

def iterable_to_csv_single_line_with_escape(data_iterable:Iterable, delimiter='|'):
    def escape_value(value, delimiter):
        # Convert the value to a string
        value_str = str(value)
        # Check if the value contains the delimiter or special characters
        if delimiter in value_str or '"' in value_str or '\n' in value_str:
            # Escape double quotes by doubling them
            value_str = value_str.replace('"', '""')
            # Wrap the value in double quotes
            value_str = f'"{value_str}"'
        return value_str
    return delimiter.join([escape_value(i, delimiter) for i in data_iterable])

def csv_to_innerlist_single_line_with_escape(csv_string:str, delimiter='|')->list:
    return next(csv.reader([csv_string], delimiter=delimiter, quotechar='"'))
def csv_to_dict_single_line_with_escape(keys,csv_string:str, delimiter='|')->dict:
    return dict(zip(keys, csv_to_innerlist_single_line_with_escape(csv_string, delimiter=delimiter)))


   
    
