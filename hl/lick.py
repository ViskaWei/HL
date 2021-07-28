import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm


class Fisher(object):
    """
    Fisher matrix class.
    """
    def __init__(self, wave, flux, mask, start=4500, end=9500, CO=False, test=False):
        super().__init__()
        self.wave = None
        self.wvln = None
        self.wvgp = 500
        self.wvtk = None
        self.wvtklb = None

        self.flux = None
        self.mask = None
        self.lick = None
        self.lick_color = None

        self.lp = None
        self.lpFe = None

        self.test = test

        self.vF = np.array([-2.5, -2.25, -2., -1.75, -1.5,
                            -1.25, -1., -0.75, -0.5, -0.25,  
                            0., 0.25, 0.5, 0.75])

        self.v

        self.init(wave, flux, mask, start, end, CO)

    def init(self, wave, flux, mask, start, end, CO):
        if not CO: 
            flux = flux[..., 3, 1, :]
            mask = mask[..., 3, 1]
        if self.test:
            flux = flux[2:6, 10:16, 0:4]
            mask = mask[2:6, 10:16 ,0:4]

        start_idx = np.digitize(start, wave)
        end_idx = np.digitize(end, wave)
        self.wave = wave[start_idx:end_idx]
        self.wvln = len(self.wave)
        n = (end-start)//self.wvgp
        self.wvtklb = np.array([start + self.wvgp*i for i in range(n)])
        self.wvtk = np.digitize(self.wvtklb, self.wave)

        self.flux = flux[..., start_idx:end_idx]
        self.mask = mask

        self.flux[~self.mask] = 0.0
        self.flux = 1.0 - self.flux

        print(f"Wave {start}-{end} ({self.wvln}, ) | Flux {self.flux.shape} | Mask {np.sum(~self.mask)}")

        self.get_lick()
        self.plot_lick()

    def get_laplace(self, flux=None, mask=None):
        flux = flux or self.flux
        mask = mask or self.mask
        d = len(flux.shape)
        # flux[~mask] = 0.0
        MID  = flux[1:-1, 1:-1, 1:-1]
        MID_MASK = mask[1:-1, 1:-1, 1:-1]
        U, D = flux[ :-2, 1:-1, 1:-1], flux[2:  , 1:-1, 1:-1]
        S, C = flux[1:-1,  :-2, 1:-1], flux[1:-1, 2:  , 1:-1]
        T, B = flux[1:-1,  :-2,  :-2], flux[1:-1, 2:  , 2:  ]
        sums = (U + D + S + C + T + B)
        laplace = sums / np.round(sums) -  MID
        laplace = abs(np.divide(laplace, MID))
        laplace[~MID_MASK] = 0.0  # Set NaN to zero
        self.lp = laplace
        # return laplace

    # def get_laplace(self, flux=None, mask=None):
    #     flux = flux or self.flux
    #     mask = mask or self.mask
    #     d = len(flux.shape)



    def get_lpFe(self, norm=True):
        if self.lp is None:
            self.get_laplace()

        lp  = self.lp.reshape(self.lp.shape[0], -1, self.wvln)
        lpFe = lp.mean(axis = 1)

        if norm:
            lpFe = lpFe / lpFe.std(axis=1)[:, None]
            # lpFe = np.divide(lpFe, np.expand_dims(lpFe.std(axis=1), -1))
        self.lpFe = lpFe

    def get_lpLg(self, norm=True):
        if self.lp is None:
            self.get_laplace()

        lp = self.lp.reshape(-1, self.lp.shape[2], self.wvln)
        lpLg = lp.mean(axis = 0)

        if norm:
            lpLg = lpLg / lpLg.std(axis=1)[:, None]
        self.lpLg = lpLg


    def plot_lp(self, s, ax=None, mR=0.6, label=None):
        s = s + 1
        vmin, vmax = np.quantile(s, mR), np.quantile(s, 0.95)
        print(vmin, vmax)
        ax = ax or plt.gca()
        ax.matshow(s, norm = LogNorm(), aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)

        self.set_wv_ticks(ax)

    def plot_Fe_lp(self):
        if self.lpFe is None:
            self.get_lpFe()
        f, (ax0, ax1) = plt.subplots(2, 1, figsize=(16,6), facecolor='w')
        self.plot_lp(self.lpFe, ax0, mR=0.6)
        self.plot_lick(ax1)


    def set_yticks(self, para):
        if para == "FeH":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_yticklabels(self.vF)
        elif para == "Logg":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_yticklabels(self.vL)

        ax.set_ylabel(para)





    

    def get_lick(self):
        dBands = {}
        dBands['CN'] = [[4142,4177]]
        dBands['Ca'] = [[3899, 4003], [4222, 4235], [4452, 4475], [8484, 8513],
                        [8522, 8562], [8642, 8682], [6358, 6402], [6775, 6900]]
        dBands['Fe'] = [[4369, 4420], [4514, 4559], [4634, 4720], 
                        [4978, 5054], [5246, 5286], [5312, 5363], 
                        [5388, 5415], [5697, 5720], [5777, 5797]]
        dBands['G']  = [[4281, 4316]]
        dBands['H']  = [[4839, 4877], [4084, 4122], [4320, 4364]]
        dBands['Mg'] = [[4761, 4799], [5069, 5134], [5154, 5197]]
        dBands['Na'] = [[8164, 8229], [8180, 8200], [5877, 5909]]
        dBands['Ti'] = [[6190, 6272], [6600, 6723], [5937, 5994], 
                        [7124, 7163], [7643, 7717], [5445, 5600], [4759, 4800]]

        cBands = {}
        cBands['CN'] = 'darkblue'
        cBands['Ca'] = 'red'
        cBands['Fe'] = 'yellow'
        cBands['G']  = 'purple'
        cBands['H']  = 'cyan'
        cBands['Mg'] = 'pink'
        cBands['Na'] = 'orange'
        cBands['Ti'] = 'lime'
        self.lick = dBands
        self.lick_color = cBands

    def plot_lick(self, ax=None):
        ax = ax or plt.subplots(figsize=(16,4))[1]
        # ax.grid(True)
        ax.set_ylim(0, 1)
        l_max = 1
        for idx, (key, vals) in enumerate(self.lick.items()):
            for val in vals:
                val_idx = np.digitize(val, self.wave)
                ax.fill_between(val_idx, 0, l_max, color = self.lick_color[key], label = key)
    #     for idx, (key, vals) in enumerate(lines.items()):
    #         ax.vlines(vals, -l_max, 0, color = next(color), label = key, linewidth = 2)

        self.set_unique_legend(ax)
        self.set_wv_ticks(ax, lim=True)
        ax.set_ylabel('LICK')


    def set_unique_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())


    def set_wv_ticks(self, ax, lim=False, rot=True):
        ax.set_xticks(self.wvtk)   
        ax.set_xticklabels(self.wvtklb)
        if lim:
            ax.set_xlim(0, self.wvln)
        if rot:
            ax.tick_params(axis='x', rotation=45)
        
