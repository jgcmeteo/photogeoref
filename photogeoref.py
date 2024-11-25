#!/usr/bin/env python3
###############################################################################
# $Id$
#
# Project:  Photogeoreferencing
# Purpose:  Script to georeference oblique terrestrial photographs to a DEM
# Author:   Javier G. Corripio jgc@meteoexploration.com
#
###############################################################################
# Copyright (c) 2004-2024, Javier G. Corripio jgc@meteoexploration.com
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
###############################################################################

import os
import sys
import argparse
import copy
import tkinter as tk
import yaml
import json
import numpy as np
from numpy import linalg as LA
import cv2
import urllib.request
from osgeo import gdal
import time
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure


"""
tkinter GUI to georeference a photograph to a digital elevation model (DEM)

All arguments are passed through the GUI
Full information in attached user manual
It creates or appeends an additional file georefGCP.log in the same directory as the photograph
"""

# read arguments and set defaults
usage = """
tkinter GUI to georeference a photograph to a digital elevation model (DEM).
All settings are in a yaml file, default filename is georefsettings.yml.
Calling the script with the option -s or --settings will load a different settings file.
"""
parser = argparse.ArgumentParser(description = usage)
parser.add_argument("-s", "--settings", default="georefsettings.yml", help="Full path to yaml settings file")
args = parser.parse_args()

yamlfile = args.settings

def load_settings():
    global logfgcp, outfile, geotransform, geoproj
    global elevation, vis, cv_img, cv_imgr, gcp, dirnameimg, rootbnameimg
    global nrows, ncols, x0, dx, dxdy, y1, dydx, dy, x1, y0, countp
    countp = 0 #display GCP on DEM only once
    yamlfile = yamlfilevar.get()
    with open(yamlfile, 'r') as geoset:
        grfyml =  yaml.safe_load(geoset)

    demfname = grfyml['demfname']
    visfname = grfyml['visfname']
    imgfname = grfyml['imgfname']
    GCPfname = grfyml['GCPfname']
    demfnametk.set(demfname)
    visfnametk.set(visfname)
    imgfnametk.set(imgfname)
    GCPfnametk.set(GCPfname)
    obscoords = np.array(grfyml['obscoords'])
    obscoordsXtk.set(obscoords[0])
    obscoordsYtk.set(obscoords[1])
    obscoordsZtk.set(obscoords[2])
    tgtcoords = np.array(grfyml['tgtcoords'])
    tgtcoordsXtk.set(tgtcoords[0])
    tgtcoordsYtk.set(tgtcoords[1])
    tgtcoordsZtk.set(tgtcoords[2])
    fwidthtk.set(grfyml['fwidth'])
    fheighttk.set(grfyml['fheight'])
    focallengthtk.set(grfyml['focallength'])
    rolldegtk.set(grfyml['roll'])
    dsdem = gdal.Open(demfname)
    band = dsdem.GetRasterBand(1)
    elevation = band.ReadAsArray()
    nrows, ncols = elevation.shape
    geotransform = dsdem.GetGeoTransform()
    geoproj = dsdem.GetProjection()
    x0, dx, dxdy, y1, dydx, dy = dsdem.GetGeoTransform()
    x1 = x0 + dx * ncols
    y0 = y1 + dy * nrows
    # python matrices are row,column [y,x]  
    # IDl and fortran are column,row [x,y] original development for IDL, keep it consistent
    elevation = elevation[::-1, ...]
    # or elevation = np.flipud(elevation)
    visibility = gdal.Open(visfname)
    bandv = visibility.GetRasterBand(1)
    vis = bandv.ReadAsArray()
    vis = vis[::-1, ...]
    gcp = np.genfromtxt(GCPfname, dtype={'names':('X','Y','Z','desc'),'formats':('f','f','f','a50')}, delimiter=",")
    dirnameimg = os.path.dirname(imgfname)
    bnameimg = os.path.basename(imgfname)
    rootbnameimg = os.path.splitext(bnameimg)[0]
    logfile = os.path.join(dirnameimg, 'georefGCP.log')
    logfgcp = open(logfile, 'a+')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), yamlfile, "\n",json.dumps(grfyml, indent=4), file=logfgcp, flush=True)
    cv_img = cv2.cvtColor(cv2.imread(imgfname), cv2.COLOR_BGR2RGB)
    realsizey = cv_img.shape[0]
    realsizex = cv_img.shape[1]
    # redefine this properly with only one canvas
    canvimgX = canvas.winfo_reqwidth()
    canvimgY = round(realsizey*canvX/realsizex)
    print('Procesing ',imgfname,realsizex,realsizey,canvimgX,canvimgY)
    cv_imgr = cv2.resize(cv_img,(canvimgX,canvimgY))
    ax.clear()
    ax.imshow(cv_imgr)
    canvasmp.draw()
    button_gcp['state'] = tk.NORMAL
    button_accept['state'] = tk.NORMAL


def writeTIFF(filename,geotransform,geoprojection,data):
    (x,y) = data.shape
    data = data[::-1, ...]
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_datatype = gdal.GDT_Float32
    dst_nbands = 1
    dst_ds = driver.Create(filename,y,x,dst_nbands,dst_datatype)
    dst_ds.GetRasterBand(1).WriteArray(data[:,:])
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)


def writeTIFFRGB(filename,geotransform,geoprojection,data):
    (x,y,z) = data.shape
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_datatype = gdal.GDT_Byte
    dst_ds = driver.Create(filename,y,x,z,dst_datatype)
    for i in range(z):  
        #bi = z-i     # Band order cv2 != geotiff not the case now.
        bi = i+1
        dst_ds.GetRasterBand(bi).WriteArray(data[:,:,i])

    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    dst_ds.FlushCache() 



def get_projmatrices(obscoords,tgtcoords,upperleftx,y0,focallength,fwidth,fheight,rolldeg):
    global logfgcp
    d = focallength
    w = fwidth/2.0
    h = fheight/2.0
    vectorc = copy.copy(obscoords)
    vectorc[0] = vectorc[0] - upperleftx
    vectorc[1] = vectorc[1] - y0
    # viewing direction vector
    vectorview = tgtcoords - obscoords
    # return('a','b')
    # vectorn = vectorview/np.sqrt(np.sum(vectorview**2))  # unit vector
    vectorn = vectorview / LA.norm(vectorview)
    print('vectorn', vectorn, file=logfgcp)
    # projection of vectorn on the xy plane (vector u will be the  crossproduct
    # of np(xy) x n (nproyection on plane xy)and therefore with 0 z-component
    vectornp = copy.copy(vectorn)
    vectornp[2] = 0
    # this avoids error for a perfectly horizontal vectorn, n == np
    vectorncross = copy.copy(vectorn)
    if vectorncross[2] == 0:
        vectorncross[2] = 1

    # to avoid inversion of projection when looking downwards:
    if vectorn[2] > 0:
        vectoru = np.cross(vectornp, vectorncross)
    else:
        vectoru = np.cross(vectorncross, vectornp)

    vectoru = vectoru / LA.norm(vectoru)  # unit vector
    print('vectoru', vectoru, file=logfgcp)
    # correction for roll of camera
    rolladd = np.tan(np.radians(-rolldeg))
    vectoru[2] = vectoru[2] + rolladd
    # unit vector
    vectoru = vectoru / LA.norm(vectoru)
    # vector v is crossproduct u x n
    vectorv = np.cross(vectoru, vectorn)
    # ttrans translation matrix to camera origin of coordinates
    ttrans = np.matrix([[1, 0, 0, -vectorc[0]],
                        [0, 1, 0, -vectorc[1]],
                        [0, 0, 1, -vectorc[2]],
                        [0, 0, 0, 1]])
    print("translation matrix \n", ttrans, file=logfgcp)
    # transformation matrix into camera coordinate system
    tview = np.matrix([[vectoru[0], vectoru[1], vectoru[2], 0],
                       [vectorv[0], vectorv[1], vectorv[2], 0],
                       [vectorn[0], vectorn[1], vectorn[2], 0],
                       [0, 0, 1 / d, 1]])
    print("transformation matrix tview \n", tview, file=logfgcp)

    # viewing direction, field of view
    vazimuth = np.rad2deg(np.arctan2(vectorn[0],vectorn[1]))
    velevation = np.rad2deg(np.arcsin(vectorn[2]))
    viewdir = ([vazimuth],[velevation])
    FOVx = np.rad2deg(np.arctan(w/d)) # film semiaxis x / focal length
    FOVy = np.rad2deg(np.arctan(h/d)) # film semiaxis y / focal length
    print('Viewing direction', viewdir, file=logfgcp)
    print('1/2 Field of view X', FOVx, file=logfgcp)
    print('1/2 Field of view Y', FOVy, file=logfgcp)

    return(ttrans,tview)


def process_gcp(display=1):
    global elevation, vis, cv_img, gcp, dirnameimg, rootbnameimg, geotransform, geoproj
    global nrows, ncols, x0, dx, dxdy, y1, dydx, dy, x1, y0
    global imgx, imgy, countp
    button_accept['state'] = tk.DISABLED
    outcoimg = os.path.join(dirnameimg,  rootbnameimg + 'coimg.jpg')
    demgcpimg = os.path.join(dirnameimg, rootbnameimg + 'DEMGCP.jpg')
    outdempersp = os.path.join(dirnameimg,'demperspective.png')
    #get variable values, they can change before every call
    obscoordsX  = obscoordsXtk.get()
    obscoordsY  = obscoordsYtk.get()
    obscoordsZ  = obscoordsZtk.get()
    tgtcoordsX  = tgtcoordsXtk.get()
    tgtcoordsY  = tgtcoordsYtk.get()
    tgtcoordsZ  = tgtcoordsZtk.get()
    fwidth      = fwidthtk.get()
    fheight     = fheighttk.get()
    focallength = focallengthtk.get()
    rolldeg     = rolldegtk.get()
    print('observer coordinates', (obscoordsX,obscoordsY,obscoordsZ), file=logfgcp)
    print('target coordinates', (tgtcoordsX,tgtcoordsY,tgtcoordsZ), file=logfgcp)
    print('rolldeg', rolldeg, file=logfgcp)
    z = elevation
    coordsc = np.zeros([nrows, ncols, 4])
    point = np.zeros([nrows, ncols, 4])
    # yscl = wysize / nrows
    upperleftx = x0
    upperlefty = y1
    dl = dx
    # observer & target position relative to file
    io = np.rint((obscoordsX - upperleftx) / dl)
    jo = np.rint((obscoordsY - y0) / dl)
    it = np.rint((tgtcoordsX - upperleftx) / dl)
    jt = np.rint((tgtcoordsY - y0) / dl)
    # ground control points, deal with nans and points outside the domain
    gcpxna = np.where(np.isnan(gcp[:]['X']))
    gcpyna = np.where(np.isnan(gcp[:]['Y']))
    gcp = np.nan_to_num(gcp)
    numgcp = gcp.shape[0]
    pointgcp = np.zeros((numgcp,3))
    pointgcp[:,0] = (gcp[:]['X']-upperleftx)/dl
    pointgcp[:,1] = (gcp[:]['Y']-y0)/dl
    pointgcp[:,2] = gcp[:]['Z']
    pointgcp = pointgcp[np.where(np.logical_and(pointgcp[:,0] >= 0, pointgcp[:,0] <= upperleftx))]
    pointgcp = pointgcp[np.where(np.logical_and(pointgcp[:,1] >= 0, pointgcp[:,1] <= upperlefty))]
    d = focallength
    w = fwidth/2.0
    h = fheight/2.0
    realsizey = cv_img.shape[0]
    realsizex = cv_img.shape[1]
    resolution = realsizex/fwidth/100.0
    # scale = 1/2 width of film(w)* 100 (m to cm) *(resolution in dpcm)
    scale = 100.0 * w * resolution
    widthheight = canvasmp.get_width_height()
    width = widthheight[0]
    height = widthheight[1]
    sclx = (2./3.)*width/realsizex
    scly = sclx
    obscoords = np.array([obscoordsX,obscoordsY,obscoordsZ])
    tgtcoords = np.array([tgtcoordsX,tgtcoordsY,tgtcoordsZ])
    ttrans,tview = get_projmatrices(obscoords,tgtcoords,upperleftx,y0,focallength,fwidth,fheight,rolldeg)
     # matrix multiplication of whole layers
    onenrows = np.ones(nrows)
    seqnrows = np.r_[0:nrows]
    seqncols = np.r_[0:ncols]
    onencols = np.ones(ncols)
    layerx = np.outer(onenrows, seqncols) * dl + ttrans[0, 3]
    layery = np.outer(seqnrows, onencols) * dl + ttrans[1, 3]
    layerz = z + ttrans[2, 3]
    # Set visibility to zero if points beside the camera.
    # if dot product (vectorview . translated_dem) < 0 => angle obtuse, point beside cam
    # this is not true for fisheye lenses, but those are not recommended as distortion would be too big
    vectorview = tgtcoords - obscoords
    visproj = layerx*vectorview[0] + layery*vectorview[1] + layerz*vectorview[2]
    vis[visproj < 0] = 0
    layerx = layerx * vis
    layery = layery * vis
    layerz = layerz * vis 
    # add correction for Earth curvature
    # good diagram: http://walter.bislins.ch/bloge/index.asp?page=Rainy+Lake+Experiment%3A+Equations#H_Drop_of_Surface
    R = 6371000  # Earth radius
    disth = np.sqrt(layerx**2 + layery**2)
    drop = disth**2/(2*R)
    layerz = layerz - drop
    layerw = np.ones([nrows, ncols])
    coordsc[:, :, 0] = tview[0, 0] * layerx + tview[0, 1] * \
        layery + tview[0, 2] * layerz + tview[0, 3] * layerw
    coordsc[:, :, 1] = tview[1, 0] * layerx + tview[1, 1] * \
        layery + tview[1, 2] * layerz + tview[1, 3] * layerw
    coordsc[:, :, 2] = tview[2, 0] * layerx + tview[2, 1] * \
        layery + tview[2, 2] * layerz + tview[2, 3] * layerw
    coordsc[:, :, 3] = tview[3, 0] * layerx + tview[3, 1] * \
        layery + tview[3, 2] * layerz + tview[3, 3] * layerw
    imgx = coordsc[:, :, 0] 
    imgy = coordsc[:, :, 1] 
    imgz = coordsc[:, :, 2] 
    # perspective projection
    imgw = w * imgz / d
    with np.errstate(invalid='ignore'):
        imgx = imgx / imgw
        imgy = imgy / imgw

    # midpoint of nominal size of photography
    midx = np.rint(realsizex / 2.0)
    midy = np.rint(realsizey / 2.0)
    # bring midpoint of projection  to midpoint of image
    imgx = np.rint(scale * imgx + midx)
    imgy = np.rint(scale * imgy + midy)
    imgx = np.nan_to_num(imgx)
    imgy = np.nan_to_num(imgy)
    imgx = np.clip(imgx, 0, realsizex-1)
    imgy = np.clip(imgy, 0, realsizey-1)
    imgx = imgx * vis
    imgy = imgy * vis
    nullx = np.where(imgx <= 0)
    nully = np.where(imgy <= 0)
    imgx[nully] = 0
    imgy[nullx] = 0
    imgx = imgx.astype(int)
    imgy = imgy.astype(int)
    # coimg  DEM as blue dots
    coimg = copy.copy(cv_img)
    coimg = coimg[::-1, ...]
    coimg[[imgy], [imgx]] = [0, 0, 255]
    dempersp = copy.copy(cv_img[:,:,0])
    dempersp = dempersp * 0.0
    dempersp = dempersp[::-1, ...]
    dempersp[[imgy], [imgx]] = [255]
    dempersp = dempersp[::-1, ...]
    cv2.imwrite(outdempersp, dempersp )
    # GCPs  viewing transformation
    numpointgcp = pointgcp.shape[0] # valid GCP
    displaygcp = np.zeros((numgcp,4))
    for i in range(0,numpointgcp):
        point = [pointgcp[i,0]*dl,pointgcp[i,1]*dl,pointgcp[i,2],1]
        point = point + ttrans[:,3].transpose()
        displaygcp[i,:] =  tview.dot(point.transpose()).reshape(4)

    # perspective projection of GCPs
    imggcpx = displaygcp[:,0]
    imggcpy = displaygcp[:,1]
    imggcpz = displaygcp[:,2]
    imggcpw = w * imggcpz / d
    with np.errstate(invalid='ignore'):
        imggcpx = imggcpx / imggcpw
        imggcpy = imggcpy / imggcpw

    # midpoint of nominal size of photography
    midx = np.rint(realsizex / 2.0)
    midy = np.rint(realsizey / 2.0)
    # bring midpoint of projection  to midpoint of image
    imggcpx = np.rint(scale * imggcpx + midx)
    imggcpy = np.rint(scale * imggcpy + midy)
    imggcpx = np.nan_to_num(imggcpx)
    imggcpy = np.nan_to_num(imggcpy)
    imggcpx = np.clip(imggcpx, 10, realsizex-10)
    imggcpy = np.clip(imggcpy, 10, realsizey-10)
    nullx = np.where(imggcpx <= 0)
    nully = np.where(imggcpy <= 0)
    imggcpx[nully] = 0
    imggcpy[nullx] = 0
    imggcpx = imggcpx.astype(int)
    imggcpy = imggcpy.astype(int)
    # coimg  DEM as blue points
    for i in range(0,numpointgcp):
        coimg[imggcpy[i], imggcpx[i]-10:imggcpx[i]+10] = [255, 0, 0]
        coimg[imggcpy[i]-10:imggcpy[i]+10, imggcpx[i]] = [255, 0, 0]    


    coimg = coimg[::-1, ...]
    cv2.imwrite(outcoimg, cv2.cvtColor(coimg, cv2.COLOR_RGB2BGR))
    if display:
        button_wait['state'] = tk.NORMAL
        ax.axis("on")
        ax.clear()
        if countp == 0:
            ax.imshow(elevation,cmap='gray',origin='lower')
            ax.scatter(io,jo,c='b',s=40)
            ax.scatter(it,jt,c='g',s=40)
            ax.scatter(pointgcp[:,0],pointgcp[:,1],c='black',s=2)
            fig.savefig(demgcpimg, bbox_inches='tight')
            canvasmp.draw()
            button_wait.wait_variable(varwait)
            ax.imshow(cv2.resize(dempersp, (0,0), fx=sclx, fy=sclx))
            canvasmp.draw()
            ax.clear()
            button_wait.wait_variable(varwait)
        ax.imshow(cv2.resize(coimg, (0,0), fx=sclx, fy=sclx))
        canvasmp.draw()
        button_wait['state'] = tk.DISABLED
        button_accept['state'] = tk.NORMAL
        countp+=1




def write_settings():
    #write values after processing
    global dirnameimg, rootbnameimg
    timestamp = time.strftime("%Y%m%d%H%M", time.localtime())
    outyaml = yamlfilevar.get()
    bnameyaml = os.path.basename(outyaml)
    rootbnameyaml = os.path.splitext(bnameyaml)[0]
    outfilesettings = os.path.join(dirnameimg, rootbnameyaml + timestamp + '.yml')
    print('write_settings to file ',outfilesettings)
    demfname = demfnametk.get()
    visfname = visfnametk.get()
    imgfname = imgfnametk.get()
    GCPfname = GCPfnametk.get()    
    obscoordsX  = obscoordsXtk.get()
    obscoordsY  = obscoordsYtk.get()
    obscoordsZ  = obscoordsZtk.get()
    tgtcoordsX  = tgtcoordsXtk.get()
    tgtcoordsY  = tgtcoordsYtk.get()
    tgtcoordsZ  = tgtcoordsZtk.get()
    fwidth      = fwidthtk.get()
    fheight     = fheighttk.get()
    focallength = focallengthtk.get()
    rolldeg     = rolldegtk.get()
    datayml = dict(
        demfname = demfname,
        visfname = visfname,
        imgfname = imgfname,
        GCPfname = GCPfname,
        obscoords = [obscoordsX,obscoordsY,obscoordsZ],
        tgtcoords = [tgtcoordsX,tgtcoordsY,tgtcoordsZ],
        fwidth = fwidth,
        fheight= fheight,
        focallength = focallength,
        roll = rolldeg        
        )
    with open(outfilesettings, 'w') as outfilesett:
        yaml.dump(datayml, outfilesett, default_flow_style=False, sort_keys=False )



def process_all():
    print("process_all")
    global nrows, ncols, elevation, cv_img, imgx, imgy
    global dirnameimg, rootbnameimg, geotransform, geoproj
    try:
        imgx
    except NameError:
        print("Assuming all parameters correct, calling process_gcp()")
        process_gcp(display=0)
    else:
        print("writing georeferenced image...")
    outfile = os.path.join(dirnameimg, rootbnameimg + 'ref.tif')
    outfileplot = os.path.join(dirnameimg, rootbnameimg + 'refplot.png')
    cv_img = cv_img[::-1, ...]
    realsizey = cv_img.shape[0]
    realsizex = cv_img.shape[1]
    cv_img[0,:,:] = 0
    cv_img[:,0,:] = 0
    cv_img[:,realsizex-1,:] = 0
    cv_img[realsizey-1,:,:] = 0
    cv_img[:,realsizex-2,:] = 0
    zalbedo = cv_img[imgy,imgx,:]
    print('Writing file ',outfile)
    writeTIFFRGB(outfile,geotransform,geoproj,zalbedo[::-1, ...])
    write_settings()
    ax.axis("on")
    ax.clear()
    ax.imshow(zalbedo,cmap='gray',origin='lower')
    levels = np.arange(np.floor(np.amin(elevation)/100)*100, np.ceil(np.amax(elevation)/100)*100,100)
    ax.contour(elevation,colors='#A0522D',levels=levels)
    levels = np.arange(np.floor(np.amin(elevation)/100)*100, np.ceil(np.amax(elevation)/100)*100,20)
    ax.contour(elevation,colors='#A0522D',levels=levels,linewidths=0.75)
    fig.savefig(outfileplot, bbox_inches='tight')
    canvasmp.draw()


def quitall():
    window.destroy
    exit()


window = tk.Tk()
window.title("Photogeoref")
#get screen dimensions
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
dpi = window.winfo_fpixels('1i')
# make canvas 55% screen dimensions
canvX = screen_width*0.6
canvY = screen_height*0.6
figXinches = canvX/dpi
figYinches = canvY/dpi
canvas = tk.Canvas(window, width=canvX, height=canvY, background='white')
canvas.grid(row=0,column=0)

frame = tk.Frame(window)
frame.grid(row=0,column=1, sticky="n")

yamlfilevar = tk.StringVar()
demfnametk = tk.StringVar()
visfnametk = tk.StringVar()
imgfnametk = tk.StringVar()
GCPfnametk = tk.StringVar()
obscoordsXtk = tk.DoubleVar()
obscoordsYtk = tk.DoubleVar()
obscoordsZtk = tk.DoubleVar()
tgtcoordsXtk = tk.DoubleVar()
tgtcoordsYtk = tk.DoubleVar()
tgtcoordsZtk = tk.DoubleVar()
fwidthtk = tk.DoubleVar()
fheighttk = tk.DoubleVar()
focallengthtk = tk.DoubleVar()
rolldegtk = tk.DoubleVar()

label1 = tk.Label(frame, text="Georeferencing Oblique Photography").grid(row=0,column=0,columnspan=2,sticky="nw",padx=5,pady=4)
settings_label    = tk.Label(frame,text="YAML settings:").grid(row=1,column=0,sticky="w", padx=5, pady=4)
demfname_label    = tk.Label(frame,text="DEM full path:").grid(row=3,column=0,sticky="w", padx=5, pady=4)
visfname_label    = tk.Label(frame,text="Viewshed full path:").grid(row=4,column=0,sticky="w", padx=5, pady=4)
imgfname_label    = tk.Label(frame,text="Image full path:").grid(row=5,column=0,sticky="w", padx=5, pady=4)
GCPfname_label    = tk.Label(frame,text="GCP full path:").grid(row=6,column=0,sticky="w", padx=5, pady=4)
obscoords_label   = tk.Label(frame,text="Observer Coordinates:").grid(row=7,column=0,columnspan=2, sticky="w", padx=5, pady=4)
obscoordsX_label  = tk.Label(frame,text="X").grid(row=8,column=0, sticky="e", padx=5, pady=4)
obscoordsY_label  = tk.Label(frame,text="Y").grid(row=9,column=0, sticky="e", padx=5, pady=4)
obscoordsZ_label  = tk.Label(frame,text="Z").grid(row=10,column=0, sticky="e", padx=5, pady=4)
tgtcoords_label   = tk.Label(frame,text="Target Coordinates:").grid(row=11,column=0, sticky="w", padx=5, pady=4)
tgtcoordsX_label  = tk.Label(frame,text="X").grid(row=12,column=0, sticky="e", padx=5, pady=4)
tgtcoordsY_label  = tk.Label(frame,text="Y").grid(row=13,column=0, sticky="e", padx=5, pady=4)
tgtcoordsZ_label  = tk.Label(frame,text="Z").grid(row=14,column=0, sticky="e", padx=5, pady=4)
fwidth_label      = tk.Label(frame,text="Sensor width [m]:").grid(row=15,column=0, sticky="w", padx=5, pady=4)
fheight_label     = tk.Label(frame,text="Sensor height [m]:").grid(row=16,column=0, sticky="w", padx=5, pady=4)
focallength_label = tk.Label(frame,text="Focal length [m]:").grid(row=17,column=0, sticky="w", padx=5, pady=4)
rolldeg_label     = tk.Label(frame,text="roll [\u00B0]:").grid(row=18,column=0, sticky="w", padx=5, pady=4)

yamlfile_entry = tk.Entry(frame,textvariable=yamlfilevar,width=50).grid(row = 1,column = 1,sticky ='w', padx=5, pady=4)
yamlfilevar.set(yamlfile)
button_loadsettings = tk.Button(frame,text="Load Settings",command=load_settings).grid(row = 2,column = 1, sticky = "we", padx=5, pady=4)
demfname_entry    = tk.Entry(frame,textvariable=demfnametk,width=50,bg="lightgrey").grid(row = 3,column = 1, sticky='w', padx=5, pady=4)
visfname_entry    = tk.Entry(frame,textvariable=visfnametk,width=50,bg="lightgrey").grid(row = 4,column = 1, sticky='w', padx=5, pady=4)
imgfname_entry    = tk.Entry(frame,textvariable=imgfnametk,width=50,bg="lightgrey").grid(row = 5,column = 1,sticky ='w', padx=5, pady=4)
GCPfname_entry    = tk.Entry(frame,textvariable=GCPfnametk,width=50,bg="lightgrey").grid(row = 6,column = 1,sticky ='w', padx=5, pady=4)
obscoordsX_entry  = tk.Entry(frame,textvariable=obscoordsXtk,width=50).grid(row = 8,column = 1,sticky ='w', padx=5, pady=4)
obscoordsY_entry  = tk.Entry(frame,textvariable=obscoordsYtk,width=50).grid(row = 9,column = 1,sticky ='w', padx=5, pady=4)
obscoordsZ_entry  = tk.Entry(frame,textvariable=obscoordsZtk,width=50).grid(row = 10,column = 1,sticky ='w', padx=5, pady=4)
tgtcoordsX_entry  = tk.Entry(frame,textvariable=tgtcoordsXtk,width=50).grid(row = 12,column = 1,sticky ='w', padx=5, pady=4)
tgtcoordsY_entry  = tk.Entry(frame,textvariable=tgtcoordsYtk,width=50).grid(row = 13,column = 1,sticky ='w', padx=5, pady=4)
tgtcoordsZ_entry  = tk.Entry(frame,textvariable=tgtcoordsZtk,width=50).grid(row = 14,column = 1,sticky ='w', padx=5, pady=4)
fwidth_entry      = tk.Entry(frame,textvariable=fwidthtk,width=50).grid(row = 15,column = 1,sticky ='w', padx=5, pady=4)
fheight_entry     = tk.Entry(frame,textvariable=fheighttk,width=50).grid(row = 16,column = 1,sticky ='w', padx=5, pady=4)
focallength_entry = tk.Entry(frame,textvariable=focallengthtk,width=50).grid(row = 17,column = 1,sticky ='w', padx=5, pady=4)
rolldeg_entry     = tk.Entry(frame,textvariable=rolldegtk,width=50).grid(row = 18,column = 1,sticky ='w', padx=5, pady=4)


# logofname = "/home/jgc/python/photogeoref/meteoexplorationtr.jpg"
# cv_logo = cv2.cvtColor(cv2.imread(logofname), cv2.COLOR_BGR2RGB)
logofname = "https://meteoexploration.com/static/assets/img/meteoexplorationtr.jpg"
try:
    req = urllib.request.urlopen(logofname)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    cv_logobgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    cv_logo = cv2.cvtColor(cv_logobgr, cv2.COLOR_BGR2RGB)
    fig = Figure(figsize=(figXinches, figYinches))
    ax = fig.add_subplot()
    ax.axis("off")
    ax.imshow(cv_logo)
except:
    donothing = 'image not loaded'
    # or plot text..

canvasmp = FigureCanvasTkAgg(fig, master=canvas)  # A tk.DrawingArea.
canvasmp.draw()
toolbar = NavigationToolbar2Tk(canvasmp, window, pack_toolbar=False)
toolbar.update()
#to be sure canvas is conected
canvasmp.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvasmp.mpl_connect("key_press_event", key_press_handler)
varwait = tk.IntVar()
button_gcp = tk.Button(frame, text="Process GCP", command=process_gcp,state=tk.DISABLED)
button_gcp.grid(row = 19,column = 1, sticky = "we", padx=5, pady=4)
button_wait = tk.Button(frame, text="Next display", command=lambda: varwait.set(1),state=tk.DISABLED)
button_wait.grid(row = 20,column = 1, sticky = "e", padx=15, pady=4)
button_accept = tk.Button(frame, text="Accept", command=process_all,state=tk.DISABLED)
button_accept.grid(row = 21,column = 1, sticky = "we", padx=5, pady=4)
button_quit = tk.Button(frame, text="Quit", command=quitall).grid(row = 22,column = 1, sticky = "we", padx=5, pady=4)
toolbar.grid(row = 1,column = 0,padx=5, pady=4)
canvasmp.get_tk_widget().grid(row = 0,column = 0, sticky = "nw", padx=5, pady=4)

window.mainloop()
