import glob
import pdb
import matplotlib.pyplot as plt 
import numpy as np
import os
import tensorflow as tf
import struct
plt.style.use('seaborn')

def get_section_results_tb(file):
    data = {}
    for i in range(8):
        data[f'scalars/eval/final_success/goal_{i}'] = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in data.keys():
                [x] = struct.unpack('f', v.tensor.tensor_content)
                data[v.tag].append(x)
    return data

if __name__ == '__main__':
    logdir = ("/home/ademi_adeniji/lexastuff/experiments/*robobin*/events*")
    logdir_tokens = str(logdir).split('/')
    eventfiles = glob.glob(logdir)
    num_points = 60
    for eventfile in eventfiles:
        data = get_section_results_tb(eventfile)
        if len(data['scalars/eval/final_success/goal_0']) < num_points:
            continue
        average_success = np.array(data['scalars/eval/final_success/goal_0'][:num_points])
        for i in range(1, 8):
            average_success += np.array(data[f'scalars/eval/final_success/goal_{i}'][:num_points])
        average_success = average_success / 8
        eventfile_tokens = str(eventfile).split('/')
        steps = np.array([5100 + (i*50000) for i in range(num_points)])
        label = eventfile_tokens[-2].split("_")[3]+eventfile_tokens[-2].split("_")[4]
        plt.plot(steps, average_success, label=label)
        plt.xlabel("env steps")
        plt.ylabel("success rate")

    plt.title('lexa+dvd success rate on robobin')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()

    plot_dir = '/'.join(logdir_tokens[:-2])+'/plots'
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    save_path = plot_dir + '/' + 'lexa_dvd_success_rate_plot.png'
    plt.savefig(save_path)   