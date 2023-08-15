#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:49:05 2023

@author: user
"""

NROWS, NCOLS = 6, 2

import numpy as np
import json
import matplotlib.pyplot as plt
import re

from pathlib import Path


# %%
def load_trials(json_f):
    trials = {}

    # Iterate over each specified JSON file
    # XXX: maxST, maxLF, ref keys depending on label
    for j in json_f:
        with open(j) as file:
            trial  = json.load(file)
            matrix = trial['mtx']
            method = trial['method']
            x_id   = trial['x_id']

            if matrix not in trials:
                trials[matrix] = {}
            if x_id not in trials[matrix]:
                trials[matrix][x_id] = {}

            trials[matrix][x_id][method] = {'fre': trial['fre'], 'relres': trial['rk']}

    return trials


def plot_trial(labels, trials, matrix, x_id, x_lim, ax, y_title=None, alt_legend=False):
    for m, label in enumerate(labels, start=1):
        relres  = trials[matrix][x_id][label]['relres']
        relres += [0] * (x_lim - len(relres))
        
        fre  = trials[matrix][x_id][label]['fre']
        fre += [0] * (x_lim - len(relres))

        # XXX: trick to display method name in legend, when comparing different preconditioner types
        label_m = label if alt_legend else f'm = {m}'
        ax.plot(range(1, x_lim+1), relres[:x_lim], label=label_m)
    
    ax.legend(fontsize='8')

    if y_title is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(y_title)
        ax2.set_yticks([])
    
     
# %%
def main(json_f):
    trials = load_trials(json_f)
    x_lim = 200

    for matrix in trials.keys():
        for x_id in trials[matrix].keys():
            fig1, ax1 = plt.subplots(nrows=NROWS, ncols=NCOLS, sharey=True)
            fig1.set_size_inches(10, 16)
            fig1.set_dpi(300)

            for i in range(NROWS):
                ax1[i, 0].set_yscale('log')
                ax1[i, 1].set_yscale('log')
                ax1[i, 1].yaxis.set_tick_params(labelleft=True)
                ax1[i, 0].set_ylabel('relres')

            ax1[-1, 0].set_xlabel('iterations')
            ax1[-1, 1].set_xlabel('iterations')


            # R1, C1) Reference results
            labels11 = ['orig', 'diagp0', 'diagp1', 'diagp2', 'tridiag', 'ilu0']
            
            plot_trial(labels11, trials, matrix, x_id, x_lim, ax1[0, 0], y_title=None, alt_legend=True)


            # TODO: use plot_trial()
            # R1, C2) MST + LF
            labels12 = ['max-st', 'max-lf']
            
            plot_trial(labels12, trials, matrix, x_id, x_lim, ax1[0, 1], y_title=None, alt_legend=True)


            # R2, C1) MST + MOS-a
            labels21 = ['max-st',
                        'max-st_mos-a_m2',
                        'max-st_mos-a_m3',
                        'max-st_mos-a_m4'
            ]
            plot_trial(labels21, trials, matrix, x_id, x_lim, ax1[1, 0], 'MOS-a (MST)')


            # R3, C1) MST + MOS-d
            labels31 = ['max-st',
                        'max-st_mos-d_m2',
                        'max-st_mos-d_m3',
                        'max-st_mos-d_m4'
            ]
            plot_trial(labels31, trials, matrix, x_id, x_lim, ax1[2, 0], 'MOS-d (MST)')

            
            # R4, C1) MST + ALT-o
            labels41 = ['max-st',
                        'max-st_alt-o_m2',
                        'max-st_alt-o_m3',
                        'max-st_alt-o_m4'
            ]
            plot_trial(labels41, trials, matrix, x_id, x_lim, ax1[3, 0], 'ALT-o (MST)')


            # R5, C1) MST + ALT-i
            labels51 = ['max-st',
                        'max-st_alt-i_m2',
                        'max-st_alt-i_m3',
                        'max-st_alt-i_m4'
            ]
            plot_trial(labels51, trials, matrix, x_id, x_lim, ax1[4, 0], 'ALT-i (MST)')


            # R6, C1) MST + ALT-o-repeat
            labels61 = ['max-st',
                        'max-st_alt-o-repeat_m2',
                        'max-st_alt-o-repeat_m3',
                        'max-st_alt-o-repeat_m4'
            ]
            plot_trial(labels61, trials, matrix, x_id, x_lim, ax1[5, 0], 'ALT-o-rep (MST)')
            
            
            # # R7, C1) MST + addition
            # labels71 = ['max-st',
            #             'max-st_add_m2',
            #             'max-st_add_m3',
            #             'max-st_add_m4'
            # ]
            # plot_trial(labels71, trials, matrix, x_id, x_lim, ax1[6, 0], 'add (MST)')
 
    
            # R2, C2) LF + MOS-a
            labels22 = ['max-lf',
                        'max-lf_mos-a_m2',
                        'max-lf_mos-a_m3',
                        'max-lf_mos-a_m4'
            ]
            plot_trial(labels22, trials, matrix, x_id, x_lim, ax1[1, 1], 'MOS-a (LF)')

            
            # R3, C2) LF + MOS-d
            labels32 = ['max-lf',
                        'max-lf_mos-d_m2',
                        'max-lf_mos-d_m3',
                        'max-lf_mos-d_m4'
            ]
            plot_trial(labels32, trials, matrix, x_id, x_lim, ax1[2, 1], 'MOS-d (LF)')

            
            # R4, C2) LF + ALT-o
            labels42 = ['max-lf',
                        'max-lf_alt-o_m2',
                        'max-lf_alt-o_m3',
                        'max-lf_alt-o_m4'
            ]
            plot_trial(labels42, trials, matrix, x_id, x_lim, ax1[3, 1], 'ALT-o (LF)')

            
            # R5, C2) LF + ALT-i
            labels52 = ['max-lf',
                        'max-lf_alt-i_m2',
                        'max-lf_alt-i_m3',
                        'max-lf_alt-i_m4'
            ]
            plot_trial(labels52, trials, matrix, x_id, x_lim, ax1[4, 1], 'ALT-i (LF)')

                
            # R6, C2) LF + ALT-o-repeat
            labels62 = ['max-lf',
                        'max-lf_alt-o-repeat_m2',
                        'max-lf_alt-o-repeat_m3',
                        'max-lf_alt-o-repeat_m4'
            ]
            plot_trial(labels62, trials, matrix, x_id, x_lim, ax1[5, 1], 'ALT-o-rep (LF)')


            # # R7, C2) LF + addition
            # labels72 = ['max-lf',
            #             'max-lf_add_m2',
            #             'max-lf_add_m3',
            #             'max-lf_add_m4'
            # ]
            # plot_trial(labels72, trials, matrix, x_id, x_lim, ax1[6, 1], 'add(LF)')
            

            fig1.savefig(f'{matrix}_x{x_id}.png', bbox_inches='tight')
            plt.close()


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='mst_plot', description='plots for spanning tree preconditioner')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for random numbers')
    parser.add_argument('--max-outer', type=int, default=15, 
                        help='maximum number of outer GMRES iterations')
    parser.add_argument('--max-inner', type=int, default=20,
                        help='maximum number of inner GMRES iterations')
    parser.add_argument('json_f', nargs='+')
    
    args = parser.parse_args()
    main(args.json_f)