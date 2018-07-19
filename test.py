# Testing module
import tensorflow as tf
import os
import profile_tf as profiler
import mobilenet_v1 as mobile
import squeezenet as squeeze
import shufflenet as shuffle
import report as report
import explorer as ex

def test_mobilenet():
    mobilenet_creator = mobile.Model("mobilenet_v1")
    mg = ex.model_generator(name = 'mobilenet_profile')

    mg.set_creator(mobilenet_creator.model_creator)
    mg.set_stats_updater(mobilenet_creator.stat_updater)

    param_list = []
    stat_list = []
    for res in [1, 0.858, 0.715, 0.572]:
        for width in [1, 0.75, 0.5]:
            for depth in [1, 2]:
                param = {'resolution_multiplier': res,
                              'width_multiplier': width,
                              'depth_multiplier': depth}
                param_list.append(param)
                stat_list.append(mg.set_and_stats(param))
    param_fields = ['resolution_multiplier', 'width_multiplier',
                    'depth_multiplier']
    stat_fields = ['param','flops','single_thread_mean','single_thread_var',
                    'multi_thread_mean','multi_thread_var', 'file_size']
    report.write_data_to_csv(param_list, param_fields, 'model_parameters')
    report.write_data_to_csv(stat_list, stat_fields, 'model_behaviour')
    # Prepare for execution graph plotting
    x_label = []
    y_single_mean = []
    y_multi_mean = []
    y_single_error = []
    y_multi_error = []

    def create_label(d):
        ret = ""
        for key, value in d.items():
            if ret=="":
                ret = str(value)
            else:
                ret = ret+"_"+str(value)
        return ret

    for i in range(len(stat_list)):
        x_label.append(create_label(param_list[i]))
        y_single_mean.append(stat_list[i]['single_thread_mean'])
        y_multi_mean.append(stat_list[i]['multi_thread_mean'])
        y_single_error.append(stat_list[i]['single_thread_var'])
        y_multi_error.append(stat_list[i]['multi_thread_var'])
    name = ['single', 'multi']
    y_data_mean = [y_single_mean, y_multi_mean]
    y_data_var = [y_single_error, y_multi_error]
    report.graph_data_bar(x_label, y_data_mean, y_data_var, name, multi=True)

def test_squeezenet():
    squeezenet_creator = squeeze.Model("squeezenet")
    mg = ex.model_generator(name = 'squeezenet_profile')

    mg.set_creator(squeezenet_creator.model_creator)
    mg.set_stats_updater(squeezenet_creator.stat_updater)

    param_list = []
    stat_list = []
    for base_expand_kernels in [128]:
        for expansion_increment in [128]:
            for pct in [0.5]:
                for freq in [2]:
                    for SR in [0.125]:
                        param = {'base_expand': base_expand_kernels,
                                      'expansion_increment': expansion_increment,
                                      'expansion_filter_ratio': pct,
                                      'filter_expansion_freq': freq,
                                      'squeeze_ratio': SR}
                        param_list.append(param)
                        stat_list.append(mg.set_and_stats(param))
    param_fields = ['base_expand', 'expansion_increment',
                    'expansion_filter_ratio', 'filter_expansion_freq'
                    'squeeze_ratio']

    stat_fields = ['param','flops','single_thread_mean','single_thread_var',
                    'multi_thread_mean','multi_thread_var', 'file_size']
    report.write_data_to_csv(param_list, param_fields, 'model_parameters')
    report.write_data_to_csv(stat_list, stat_fields, 'model_behaviour')
    # Prepare for execution graph plotting
    x_label = []
    y_single_mean = []
    y_multi_mean = []
    y_single_error = []
    y_multi_error = []

    def create_label(d):
        ret = ""
        for key, value in d.items():
            if ret=="":
                ret = str(value)
            else:
                ret = ret+"_"+str(value)
        return ret

    for i in range(len(stat_list)):
        x_label.append(create_label(param_list[i]))
        y_single_mean.append(stat_list[i]['single_thread_mean'])
        y_multi_mean.append(stat_list[i]['multi_thread_mean'])
        y_single_error.append(stat_list[i]['single_thread_var'])
        y_multi_error.append(stat_list[i]['multi_thread_var'])
    name = ['single', 'multi']
    y_data_mean = [y_single_mean, y_multi_mean]
    y_data_var = [y_single_error, y_multi_error]
    report.graph_data_bar(x_label, y_data_mean, y_data_var, name, multi=True)

def test_shufflenet():
    shufflenet_creator = shuffle.Model("shufflenet")
    mg = ex.model_generator(name = 'shufflenet_profile')

    mg.set_creator(shufflenet_creator.model_creator)
    mg.set_stats_updater(shufflenet_creator.stat_updater)

    # Mapping out_channel with nr_group
    out_ch_map = {1: 144, 2: 200, 3: 240, 4: 272, 8:384}
    param_list = []
    stat_list = []
    for filter_group in [1,2,3,4,8]:
        for complexity_scale_factor in [0.25, 0.5, 1.0]:
            param = {'filter_group': filter_group,
                    'complexity_scale_factor': complexity_scale_factor,
                    'out_channel': out_ch_map[filter_group]}
            param_list.append(param)
            stat_list.append(mg.set_and_stats(param))
    param_fields = ['filter_group', 'complexity_scale_factor']

    stat_fields = ['param','flops','single_thread_mean','single_thread_var',
                    'multi_thread_mean','multi_thread_var', 'file_size']
    report.write_data_to_csv(param_list, param_fields, 'model_parameters')
    report.write_data_to_csv(stat_list, stat_fields, 'model_behaviour')
    # Prepare for execution graph plotting
    x_label = []
    y_single_mean = []
    y_multi_mean = []
    y_single_error = []
    y_multi_error = []

    def create_label(d):
        ret = ""
        for key, value in d.items():
            if ret=="":
                ret = str(value)
            else:
                ret = ret+"_"+str(value)
        return ret

    for i in range(len(stat_list)):
        x_label.append(create_label(param_list[i]))
        y_single_mean.append(stat_list[i]['single_thread_mean'])
        y_multi_mean.append(stat_list[i]['multi_thread_mean'])
        y_single_error.append(stat_list[i]['single_thread_var'])
        y_multi_error.append(stat_list[i]['multi_thread_var'])
    name = ['single', 'multi']
    y_data_mean = [y_single_mean, y_multi_mean]
    y_data_var = [y_single_error, y_multi_error]
    report.graph_data_bar(x_label, y_data_mean, y_data_var, name, multi=True)

def main():
    test_mobilenet()
    #test_squeezenet()
    #test_shufflenet()


if __name__ == "__main__":
    main()
