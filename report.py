"""
Reporter tools that includes:
    * Extracting relevant information from cmd output
    * Formating the extracted information
    * Presenting the information as:
        * CSV: raw results as described by feature vector
        * Graphs: numerical graphing with common stats such as deviation

This module is GPLv3 licensed.

"""

import numpy as np
import os
import csv

def format_adb_msg(msg):
    pass


def write_data_to_csv(data, fields, file_name):
    with open(file_name+'.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

def main():
    def report_test():
        fields = ['feature_name', 'mean', 'std']
        f1 = [45, 0.2]
        f2 = [12, 1]
        data = []
        data.append({fields[0]: 'feature1', fields[1]: f1[0], fields[2]: f1[1]})
        data.append({fields[0]: 'feature2', fields[1]: f2[0], fields[2]: f2[1]})
        write_data_to_csv(data, fields, "sample_file")

    report_test()

if __name__ == "__main__":
    main()
