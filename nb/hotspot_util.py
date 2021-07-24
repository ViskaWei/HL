from __future__ import print_function
import os
import sys
import copy
import getpass
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd 
import h5py
from matplotlib.colors import LogNorm
from ipywidgets import interact, interactive, FloatSlider,fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display


flux30 = np.load("flux3.npy")
flux3_mask = np.load("flux3_mask.npy")
grid_wv0 = np.load("grid_wv.npy")
wv_id = 10000
flux3 = flux30[:,:,:,:wv_id]
grid_wv = grid_wv0[:wv_id]
Lg_paras = np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])
Te_paras = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000,
       6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750,
       9000, 9250, 9500, 9750])
Fe_paras = np.array([-2.5 , -2.25, -2.  , -1.75, -1.5 , -1.25, -1.  , -0.75, -0.5 ,
       -0.25,  0.  ,  0.25,  0.5 ,  0.75])
para_dict = {}
para_dict["Lg"] = Lg_paras
para_dict["Te"] = Te_paras
para_dict["Fe"] = Fe_paras
wv_Nticks = 250

def get_widget(para, para_str):
    w = FloatSlider(min=para[0],max=para[-1], step=para[1]-para[0],continuous_update=False, description = para_str)  
    return w
def get_widgets(para_dict):
    ws = {}
    # print(para_dict.keys())
    for key, val in para_dict.items():
        ws[key] = get_widget(val, key)
    return ws

ws = get_widgets(para_dict)
wv_idxs =np.linspace(0,len(grid_wv)-1,len(grid_wv)//wv_Nticks+2 ).astype(int)
wv_tick = grid_wv[wv_idxs].astype(int)

def vac2air(vac):  
    air = vac/ (1.0 + 2.735182E-4 + 131.4182 / vac**2 + 2.76249E8 / vac**4)
    return np.round(air,2)
def air2vac(air):  
    sigma2 = 1.0e+8 / (air * air);
    n = 1 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2);
    return np.round(air * n,2)

def get_lineidx(line, grid_wv, air = True):
    if air:
        line = air2vac(line)
    print('wv in vac',line)
    line_idx = np.argsort(abs(grid_wv - line))[0]
    return line_idx

def plot_Fe_line(flux, para_str, para_dict,line_idx, grid_wv, idx_Fe_H =10, idx_Teff = 6, idx_log = 3, ds = 50 ,ax = plt):
    paras = para_dict[para_str]
    ax.plot( grid_wv[(line_idx-ds ): (line_idx+ ds)], flux[idx_Fe_H,idx_Teff,idx_log, (line_idx- ds ): (line_idx+ ds)], c = 'deepskyblue', label = f'{para_str}= {paras[idx_Fe_H]}')
    ax.axvline(grid_wv[line_idx], c ='g', label = f'Hb{np.round(grid_wv[line_idx])}')
    ax.legend(loc = 3)

def plot_Lg_line(flux, para_str, para_dict, line_idx, grid_wv, idx_Fe_H =10, idx_Teff = 2, idx_log = 3, ds = 30 ,ax = plt):
    paras = para_dict[para_str]
#     ax.plot( grid_wv, flux[idx_Fe_H,idx_Teff,idx_log], c = 'deepskyblue', label = f'{para_str}= {paras[idx_log]}')

    ax.plot( grid_wv[(line_idx-ds ): (line_idx+ ds)], flux[idx_Fe_H,idx_Teff,idx_log, (line_idx- ds ): (line_idx+ ds)], c = 'deepskyblue', label = f'{para_str}= {paras[idx_log]}')
    ax.axvline(grid_wv[line_idx], c ='g', label = 'Ca I L+')
#     ax.set_ylim([1000000, 2000000])
    ax.legend(loc = 3)

def plot_lp1d(Lg, Te, Fe, para_dict, flux3, flux3_mask, wv_tick, wv_Nticks, Lg_axis = True, Te_axis = False, Fe_axis = False):
    Lg_paras = para_dict["Lg"]
    Te_paras = para_dict["Te"]
    Fe_paras = para_dict["Fe"]
    idx_Te =np.where(Te_paras == Te)[0][0]
    idx_Lg =np.where(Lg_paras == Lg)[0][0]
    idx_Fe =np.where(Fe_paras == Fe)[0][0]
    idx_Te0, idx_Lg0, idx_Fe0 = slice(None), slice(None), slice(None)
    if Lg_axis and (not Te_axis) and (not Fe_axis):
        flux_id = (idx_Fe,idx_Te, idx_Lg0)
        paras = Lg_paras
        para_str = 'log(g)'
    elif (not Lg_axis) and (Te_axis) and (not Fe_axis):
        flux_id = (idx_Fe,idx_Te0, idx_Lg)
        paras = Te_paras
        para_str = 'Teff (K)'
    elif (not Lg_axis) and (not Te_axis) and (Fe_axis):
        flux_id = (idx_Fe0,idx_Te, idx_Lg)
        paras = Fe_paras
        para_str = '[Fe/H]'
    else:
        return None
    fluxH1D = flux3[flux_id]
    maskH1D = flux3_mask[flux_id]
    lp1D = get_abs_lp_frac(fluxH1D, maskH1D)
    fig = plt.figure(figsize = (20,6))
    ax = fig.add_subplot(111)
    pmin, pmax =np.quantile(lp1D, 0.50),  np.quantile(lp1D, 0.98)
    plot_lp_wv(paras, para_str,lp1D, pmin,  pmax, wv_Nticks, wv_tick,ax)
    ax.set_title(f'Absolute relative lapalcian along {para_str} direction',fontsize=18)
    return None

def plot_lp_wv(paras, para_str, mat, pmin, pmax, wv_Nticks, wv_tick, ax):
    if ax is None:
        ax = plt.gca()
    ax.matshow(mat+1e-6, norm = LogNorm(),vmin = pmin, vmax = pmax)
    ax.set_aspect('auto')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels(paras)
    ax.set_ylabel(para_str,fontsize=18)
    ax.xaxis.set_major_locator(ticker.IndexLocator(base=wv_Nticks, offset=0))    
    ax.set_xticklabels(wv_tick, rotation=45)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('wave (A)',fontsize=18)

def interact_lp_1d(para_dict, flux3, flux3_mask, wv_tick, wv_Nticks):
    f = lambda Lg, Te, Fe, Lg_axis, Te_axis, Fe_axis : plot_lp1d(Lg, Te, Fe, para_dict, flux3, flux3_mask, wv_tick, wv_Nticks, Lg_axis, Te_axis, Fe_axis)
    return f

def get_abs_lp_frac(fluxH1D, maskH1D):
    head, mid, tail = fluxH1D[:-2,:], fluxH1D[1:-1,:], fluxH1D[2:,:]
    laplace = (head + tail - 2 * mid)/2
    fluxH1D[~maskH1D] = None
    laplace[np.isnan(laplace)] = 0
    fluxH1D[np.isnan(fluxH1D)] = 0    
    laplace_frac = np.divide(laplace, fluxH1D[1:-1,:])
    return abs(laplace_frac)

f1d = interact_lp_1d(para_dict, flux3, flux3_mask, wv_tick, wv_Nticks)


####################################2D#############################
def plot_lp2d(Lg, Te, Fe, para_dict, flux3, flux3_mask, wv_tick, wv_Nticks, Lg_axis = True, Te_axis = True, Fe_axis = False):
    Lg_paras = para_dict["Lg"]
    Te_paras = para_dict["Te"]
    Fe_paras = para_dict["Fe"]
    idx_Te =np.where(Te_paras == Te)[0][0]
    idx_Lg =np.where(Lg_paras == Lg)[0][0]
    idx_Fe =np.where(Fe_paras == Fe)[0][0]    
    idx_Te0, idx_Lg0, idx_Fe0 = slice(None), slice(None), slice(None)
    paras = None
    if Lg_axis and (Te_axis) and (not Fe_axis):
        flux_id = (idx_Fe,idx_Te0, idx_Lg0)
        plot_id = [(idx_Te,idx_Lg0), (idx_Te0,idx_Lg)]
        paras = [Te_paras,Lg_paras]
        para_str = ['Teff', 'log(g)']
    elif (not Lg_axis) and (Te_axis) and (Fe_axis):
        flux_id = (idx_Fe0,idx_Te0, idx_Lg)
        plot_id = [(idx_Fe,idx_Te0), (idx_Fe0,idx_Te)]
        paras = [Fe_paras,Te_paras]
        para_str = ['[Fe/H]','Teff']
    elif (Lg_axis) and (not Te_axis) and (Fe_axis):
        flux_id = (idx_Fe0,idx_Te, idx_Lg0)
        plot_id = [(idx_Fe,idx_Lg0), (idx_Fe0,idx_Lg)]
        paras = [Fe_paras,Lg_paras]
        para_str = ['[Fe/H]', 'log(g)']
    else:
        return None
    flux2D = flux3[flux_id]
    mask2D = flux3_mask[flux_id]
    lp2D_wv, lp2D = get_abs_lp_frac2d(flux2D, mask2D)
    fig = plt.figure(figsize=(20,8))
#     fig, (ax0,ax1,ax2) = plt.subplots(1,3, figsize = (20,6))
    ax0 = fig.add_subplot(1, 2, 1)   #top and bottom left
    ax1 = fig.add_subplot(2, 2, 2)   #top right
    ax2 = fig.add_subplot(2, 2, 4) 
    ax0.matshow(lp2D+1e-6, norm = LogNorm())
    ax0.set_aspect('auto')
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax0.set_yticklabels(paras[0])
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(1))    
    ax0.set_xticklabels(paras[1])
    pmin, pmax = np.quantile(lp2D, 0.50), np.quantile(lp2D, 0.98)
    plot_lp_wv(paras[1], para_str[1],lp2D_wv[plot_id[0]], pmin,  pmax, wv_Nticks, wv_tick,ax1)
    plot_lp_wv(paras[0], para_str[0],lp2D_wv[plot_id[1]], pmin,  pmax, wv_Nticks, wv_tick,ax2)
    return None

def get_abs_lp_frac2d(fluxH2D, mask2D):
    d = len(fluxH2D.shape)
    fluxH2D[~mask2D] = None
    if d == 2:
        fluxH2D = fluxH2D[:,None]
    t,b, mid, l,r = fluxH2D[:-2, 1:-1,:], fluxH2D[2:,1:-1,:], fluxH2D[1:-1,1:-1, :], fluxH2D[1:-1,:-2, :], fluxH2D[1:-1,2:, :],
    laplace = (t+b+l+r - 4 * mid)/4
    laplace = abs(np.divide(laplace, fluxH2D[1:-1,1:-1,:]))
    laplace[np.isnan(laplace)] = 0  
    laplace95 = np.quantile(laplace, 0.95, axis = -1)
    return laplace, laplace95

def interact_lp_2d(para_dict, flux3, flux3_mask,wv_tick, wv_Nticks):
    f = lambda Lg, Te, Fe, Lg_axis, Te_axis, Fe_axis : plot_lp2d(Lg, Te, Fe, para_dict, flux3, flux3_mask, wv_tick, wv_Nticks, Lg_axis, Te_axis, Fe_axis)
    return f

f2d = interact_lp_2d(para_dict, flux3, flux3_mask,wv_tick, wv_Nticks)
##############################3d
def get_abs_lp_frac3d(fluxH3D, mask3D):
    d = len(fluxH3D.shape)
    fluxH3D[~mask3D] = None
    if d == 3:
        fluxH3D = fluxH3D[:,None]
    u,d = fluxH3D[:-2, 1:-1,1:-1], fluxH3D[2:,1:-1,1:-1]
    h = fluxH3D[1:-1,1:-1, 1:-1]
    s,c = fluxH3D[1:-1,:-2, 1:-1], fluxH3D[1:-1,2:, 1:-1]
    t,b = fluxH3D[1:-1,:-2, :-2], fluxH3D[1:-1,2:, 2:]
    laplace = (u+d+s+c+t+b - 6 * h)/6
    laplace = abs(np.divide(laplace, h))
    laplace[np.isnan(laplace)] = 0  
    laplace95 = np.quantile(laplace, 0.95, axis = -1)
#     print(np.shape(laplace), np.shape(laplace95))
    return laplace, laplace95

def interact_lp_3d(para_dict, flux3, flux3_mask):
    ws = get_widgets(para_dict)
    f = lambda Lg, Te, Fe, Lg_axis, Te_axis, Fe_axis : plot_lp3d(Lg, Te, Fe, para_dict, flux3, flux3_mask, Lg_axis, Te_axis, Fe_axis)
    return f ,ws

def plot_lp3d(Lg, Te, Fe, para_dict, flux3, flux3_mask, Lg_axis = True, Te_axis = False, Fe_axis = False):
    Lg_paras = para_dict["Lg"]
    Te_paras = para_dict["Te"]
    Fe_paras = para_dict["Fe"]
    idx_Te =np.where(Te_paras == Te)[0][0]
    idx_Lg =np.where(Lg_paras == Lg)[0][0]
    idx_Fe =np.where(Fe_paras == Fe)[0][0]
    
    idx_Te0, idx_Lg0, idx_Fe0 = slice(None), slice(None), slice(None)
    para = {}
    if (not Lg_axis) and (not Te_axis) and (Fe_axis):
        flux_id = (idx_Fe,idx_Te0, idx_Lg0)
        plot_id = [(idx_Fe,idx_Te,idx_Lg0), (idx_Fe,idx_Te0,idx_Lg)]
        para['p1'] = Te_paras
        para['p2'] = Lg_paras
    elif (Lg_axis) and (not Te_axis) and (not Fe_axis):
        flux_id = (idx_Fe0,idx_Te0, idx_Lg)
        plot_id = [(idx_Fe,idx_Te0,idx_Lg), (idx_Fe0,idx_Te,idx_Lg)]
        para['p1'] = Fe_paras
        para['p2'] = Te_paras   
    elif (not Lg_axis) and (Te_axis) and (not Fe_axis):
        flux_id = (idx_Fe0,idx_Te, idx_Lg0)
        plot_id = [(idx_Fe,idx_Te,idx_Lg0), (idx_Fe0,idx_Te,idx_Lg)]
        para['p1'] = Fe_paras
        para['p2'] = Lg_paras
    else:
        return None
    lp3d_wv, lp3d = get_abs_lp_frac3d(flux3, flux3_mask)
    pmin, pmax = np.quantile(lp3d_wv, 0.50), np.quantile(lp3d_wv, 0.98)

#     fig = plt.figure(figsize=(20,8))
    fig, (ax0,ax1,ax2) = plt.subplots(3,1, figsize = (20,8))
#     ax0 = fig.add_subplot(1, 2, 1)   #top and bottom left
#     ax1 = fig.add_subplot(2, 2, 2)   #top right
#     ax2 = fig.add_subplot(2, 2, 4) 
    ax0.matshow(lp3d[flux_id], norm = LogNorm())
    ax0.set_aspect('auto')
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax0.set_yticklabels(para['p1'])
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(1))    
    ax0.set_xticklabels(para['p2'])
    ax1.matshow(lp3d_wv[plot_id[0]],norm = LogNorm(), vmin =pmin,vmax =pmax)
    ax1.set_aspect('auto')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.set_yticklabels(para['p2'])
#     ax1.xaxis.set_major_locator(ticker.IndexLocator(base=wv_Nticks, offset=0))    
    ax1.set_xticklabels([],[])
    ax2.matshow(lp3d_wv[plot_id[1]],norm = LogNorm(), vmin =pmin,vmax =pmax)
    ax2.set_aspect('auto')
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.set_yticklabels(para['p1'])
    ax2.set_xticklabels([],[])
    return None
