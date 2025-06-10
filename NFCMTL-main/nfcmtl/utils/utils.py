import csv
import os
from typing import OrderedDict
import pandas as pd
import torch
from black.trans import defaultdict
from blib2to3.pgen2.driver import Any
from omegaconf import AnyNode, ListConfig
from omegaconf.base import ContainerMetadata, Metadata


def prepare_torch26():
    if torch.__version__ >= "2.6":
        torch.serialization.add_safe_globals([ListConfig])
        torch.serialization.add_safe_globals([ContainerMetadata])
        torch.serialization.add_safe_globals([Any])
        torch.serialization.add_safe_globals([list])
        torch.serialization.add_safe_globals([defaultdict])
        torch.serialization.add_safe_globals([dict])
        torch.serialization.add_safe_globals([int])
        torch.serialization.add_safe_globals([AnyNode])
        torch.serialization.add_safe_globals([Metadata])


def save_data_to_csv(data_dict: OrderedDict, filename: str):
    """
    Saves or appends an OrderedDict to a CSV file.
    If the file does not exist or is empty, headers (keys of data_dict) are written.
    Otherwise, data (values of data_dict) is appended as a new row.
    Values are converted to standard Python floats.

    Args:
        data_dict (OrderedDict): The OrderedDict to save. Keys are headers, values form the data row.
        filename (str): The name of the CSV file (e.g., 'output.csv').
    """
    if not filename.endswith('.csv'):
        print(f"Warning: Filename '{filename}' does not end with '.csv'. Adding it.")
        filename += '.csv'
    
    # Ensure the directory for the file exists.
    # os.path.dirname() returns '' for filenames without a path (e.g., 'output.csv').
    # os.makedirs('', exist_ok=True) is a no-op and safe.
    # For 'path/to/output.csv', it creates 'path/to' if it doesn't exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Determine if the header needs to be written.
    # This is true if the file doesn't exist, or if it exists but is empty.
    write_header = False
    if not os.path.exists(filename):
        write_header = True
    else:
        try:
            if os.path.getsize(filename) == 0: # File exists and is empty
                write_header = True
        except OSError as e:
            # If we can't get the size of an existing file (e.g., permissions, race condition),
            # log a warning. We'll proceed assuming it's not empty for header purposes.
            # The subsequent file operations might fail and be caught by the main try-except block.
            print(f"Warning: Could not determine if file '{filename}' is empty due to: {e}. Proceeding as if not empty for header writing.")

    try:
        # Open in append mode ('a'). This creates the file if it doesn't exist.
        # newline='' is important for csv.writer to prevent extra blank rows on some OS (e.g. Windows).
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if write_header:
                if not data_dict:
                    headers = []
                    print(f"Warning: data_dict is empty. Writing empty header to '{filename}'.")
                else:
                    headers = list(data_dict.keys())
                writer.writerow(headers)

            # Write the data row
            if not data_dict:
                row_data = []
                if write_header: # Avoid double warning if data_dict is empty
                    pass # Warning already printed for header
                else: # Appending empty row to existing file
                    print(f"Warning: data_dict is empty. Writing empty data row to '{filename}'.")

            else:
                # Convert values to standard Python floats for robust CSV writing
                row_data = [float(value) for value in data_dict.values()]
            writer.writerow(row_data)
        
        if write_header:
            # This message covers both new file and existing empty file scenarios.
            print(f"Successfully saved data with header to '{filename}'")
        else:
            # This message covers appending to an existing, non-empty file.
            print(f"Successfully appended data to '{filename}'")
            
    except IOError:
        # Catches errors from open() or writer.writerow() related to I/O (e.g., permissions, disk full)
        print(f"Error: Could not write to file '{filename}'")
    except ValueError as e:
        # Catches errors from float(value) if a value is not convertible
        print(f"Error: Could not convert data to float for CSV writing. Details: {e}")
    except Exception as e:
        # Catches other unexpected errors
        print(f"An unexpected error occurred: {e}")

