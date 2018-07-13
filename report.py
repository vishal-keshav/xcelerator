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

import plotly.offline as py
#import plotly.plotly as py
import plotly.graph_objs as go

def format_adb_msg(msg):
    # This is most basic version, searches for
    # only average time: xx.xxxx ms
    tag = "time:"
    for line in msg.splitlines():
        if tag in line:
            matched_line = line
            break
    # Extract the xx.xxxx part after "time:"
    words = matched_line.split()
    exec_time = float(words[2])
    return {'exec_time': exec_time}

def write_data_to_csv(data, fields, file_name):
    with open(file_name+'.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

def graph_data_bar(x_label, y_data_mean, y_data_var, name, multi=False):
    if multi == True:
        plot_data = []
        for i in range(len(name)):
            trace = go.Bar(x=x_label, y=y_data_mean[i], name=name[i],
                error_y=dict( type='data', array=y_data_var[i], visible=True))
            plot_data.append(trace)
        layout = go.Layout(barmode='group')
        fig = go.Figure(data=plot_data, layout=layout)
    else:
        trace = go.Bar(x=x_label, y=y_data_mean, name=name,
            error_y=dict( type='data', array=y_data_var, visible=True))
        plot_data = [trace]
        fig = go.Figure(data=plot_data)
    py.plot(fig, filename='bar_plot', image='png')

def main():
    def report_test():
        fields = ['feature_name', 'mean', 'std']
        f1 = [45, 0.2]
        f2 = [12, 1]
        data = []
        data.append({fields[0]: 'feature1', fields[1]: f1[0], fields[2]: f1[1]})
        data.append({fields[0]: 'feature2', fields[1]: f2[0], fields[2]: f2[1]})
        write_data_to_csv(data, fields, "sample_file")

    #report_test()
    label = ['a', 'b', 'c']
    mean = [[1,2,3],[4,2,3]]
    error = [[0.2, 0.4, 0.3], [0.9, 0.4, 0.1]]
    name = ['test1', 'test2']
    graph_data_bar(label, mean, error, name, True)

if __name__ == "__main__":
    main()
