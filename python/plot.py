#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:49:05 2023

@author: user
"""

NROWS, NCOLS = 7, 2

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


def main(json_f):
    trials = load_trials(json_f)

    for matrix in trials.keys():
        for x_id in trials[matrix].keys():
            fig1, ax1 = plt.subplots(nrows=NROWS, ncols=NCOLS, sharey=True)
            fig1.set_size_inches(10, 26)
            fig1.set_dpi(300)

            for i in range(NROWS):
                ax1[i, 0].set_yscale('log')
                ax1[i, 0].set_xlabel('iterations')
                ax1[i, 0].set_ylabel('relres')
            
                ax1[i, 1].set_yscale('log')
                ax1[i, 1].set_xlabel('iterations')
                ax1[i, 1].set_ylabel('relres')

            # R1) Reference results
            # XXX: replace jacobi with diagp0, diagp1, diagp2
            ref_trials = ['orig', 'diagp0', 'diagp1', 'diagp2', 'tridiag', 'ilu0']
            
            for k, label in enumerate(ref_trials):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[0, 0].plot(range(1, len(relres)+1), relres, label=label)
                ax1[0, 1].plot(range(1, len(relres)+1), relres, label=label)

            # R2, C1) MST + MOS-a
            trials21 = ['max-st',
                        'max-st_mos-a_m2',
                        'max-st_mos-a_m3',
                        'max-st_mos-a_m4'
            ]
            
            for k, label in enumerate(trials21):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[1, 0].plot(range(1, len(relres)+1), relres, label=label)

            # R3, C1) MST + MOS-d
            trials31 = ['max-st',
                        'max-st_mos-d_m2',
                        'max-st_mos-d_m3',
                        'max-st_mos-d_m4'
            ]

            for k, label in enumerate(trials31):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[2, 0].plot(range(1, len(relres)+1), relres, label=label)
                
            # R4, C1) MST + ALT-o
            trials41 = ['max-st',
                        'max-st_alt-o_m2',
                        'max-st_alt-o_m3',
                        'max-st_alt-o_m4'
            ]
            
            for k, label in enumerate(trials41):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[3, 0].plot(range(1, len(relres)+1), relres, label=label)
                
            # R5, C1) MST + ALT-i
            trials51 = ['max-st',
                        'max-st_alt-i_m2',
                        'max-st_alt-i_m3',
                        'max-st_alt-i_m4'
            ]
            
            for k, label in enumerate(trials51):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[4, 0].plot(range(1, len(relres)+1), relres, label=label)

            # R6, C1) MST + ALT-o-repeat
            trials61 = ['max-st',
                        'max-st_alt-o-repeat_m2',
                        'max-st_alt-o-repeat_m3',
                        'max-st_alt-o-repeat_m4'
            ]
            
            for k, label in enumerate(trials61):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[5, 0].plot(range(1, len(relres)+1), relres, label=label)
                
            # R7, C1) MST + addition
            trials71 = ['max-st',
                        'max-st_add_m2',
                        'max-st_add_m3',
                        'max-st_add_m4'
            ]

            for k, label in enumerate(trials71):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[6, 0].plot(range(1, len(relres)+1), relres, label=label)

            # R2, C2) LF + MOS-a
            trials22 = ['max-lf',
                        'max-lf_mos-a_m2',
                        'max-lf_mos-a_m3',
                        'max-lf_mos-a_m4'
            ]
            
            for k, label in enumerate(trials22):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[1, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            # R3, C2) LF + MOS-d
            trials32 = ['max-lf',
                        'max-lf_mos-d_m2',
                        'max-lf_mos-d_m3',
                        'max-lf_mos-d_m4'
            ]

            for k, label in enumerate(trials32):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[2, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            # R4, C2) LF + ALT-o
            trials42 = ['max-lf',
                        'max-lf_alt-o_m2',
                        'max-lf_alt-o_m3',
                        'max-lf_alt-o_m4'
            ]
            
            for k, label in enumerate(trials42):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[3, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            # R5, C2) LF + ALT-i
            trials52 = ['max-lf',
                        'max-lf_alt-i_m2',
                        'max-lf_alt-i_m3',
                        'max-lf_alt-i_m4'
            ]

            for k, label in enumerate(trials52):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[4, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            # R6, C2) LF + ALT-o-repeat
            trials62 = ['max-lf',
                        'max-lf_alt-o-repeat_m2',
                        'max-lf_alt-o-repeat_m3',
                        'max-lf_alt-o-repeat_m4'
            ]

            for k, label in enumerate(trials62):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[5, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            # R7, C2) LF + addition
            trials72 = ['max-lf',
                        'max-lf_add_m2',
                        'max-lf_add_m3',
                        'max-lf_add_m4'
            ]
             
            for k, label in enumerate(trials72):
                relres = trials[matrix][x_id][label]['relres']
                fre    = trials[matrix][x_id][label]['fre']
                
                ax1[6, 1].plot(range(1, len(relres)+1), relres, label=label)
                
            fig1.savefig(f'{matrix}_x{x_id}.png', bbox_inches='tight')


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