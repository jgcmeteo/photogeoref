# photogeoref
Georeferencing oblique photographs to a digital elevation model.
A tkinter GUI to georeference oblique photography to a DEM.

![demperspective](https://www.meteoexploration.com/static/assets/img/demperspF.jpg)


Georeferencing oblique photography can be an efficient complement to remote sensing, or a good alternative when high resolution satellite images are too expensive. This technique works on cloudy and overcast days and can be applied to multispectral cameras with some modifications. It works better for mountain terrain, where the viewing angle is more perpendicular. For flat terrain or very slanting views the projection distortion can be excessive. 

The tool works for any camera where focal length and sensor dimensions are well known. Precision is better with fixed focal length lenses.  It might work with smart phones pictures, but distortion could be very high. The images can be pre-processed, to highlight regions of interest or enhance any attribute, such as vegetation, water bodies, rivers, avalanches,  geomorphological or geological features, etc. It has been used to derive albedo from the snow, the snow line fluctuation, validating snow drift models, derive glacier flow, mapping avalanches, monitoring high altitude mining operations on glaciers, forestry and phenology studies, etc.

## Documentation

Full documentation and examples at: https://www.meteoexploration.com/python/photogeoref/index.html

## Overview

Runing the script will show a GUI where settings can be adjusted, the results are visualized on the display window.

`python3 photogeoref.py -h`  
&ensp;     Shows the command syntax and available options.

`python3 photogeoref.py`
&ensp;     Run script with default settings

Setings are written in a yaml file with the following structure

`cat georefsettings.yml`

```
demfname: '/full_path_to/dem.tif'
visfname: '/full_path_to/demviewshed.tif'
imgfname: '/full_path_to/RGB_image.tif'
GCPfname: '/full_path_to/x_y_z_description.csv'
obscoords:
- 728360.0
- 4762763.0
- 1465.0
tgtcoords:
- 728268.0
- 4763407.0
- 1497.0
fwidth: 0.0359
fheight: 0.024
focallength: 0.028
roll: 2.5   

```


`demfname`
&ensp;     Full path to a GeoTIFF with elevation data (DEM). $Delta$X, $DeltaY$ and Z should be in metres, DEM should be a UTM projection with squared pixels.

`visfname`
&ensp;     Full path to a GeoTIFF with the calculated DEM viewshed from the observer position. Visible values should be 1 and non-visible values zero.
A simple way to calculate viewshed is with gdal:

`gdal_viewshed -md 30000 -ox observerX -oy ObserverY -oz heightoverdem -vv 1 inputdem.tif outputviewshed.tif`

`imgfname`
&ensp;     Full path to a a RGB tiff with the photograph to be georeferenced. It should be the full image, not cropped. Images can be processed, contains annotations and modifications but dimensions should be preserved

`GCPfname`
&ensp;     Full path to a a comma-separated-values file (csv) with Ground Control Points. Its is advisable to include the observer and the target position.

`tgtcoords`
&ensp;     Array with the observer X, Y, Z coordinates (camera position). Observer coordinates should be inside the DEM domain and in the same reference system. Coordinates should be easting and northing in metres, not degrees of latitude or longitude.


`obscoords`
&ensp;     Array with the target X, Y, Z coordinates. The target is the real world position of the mid point of the photograph. Target coordinates should be inside the DEM domain and in the same reference system. Coordinates should be easting and northing in metres, not degrees of latitude or longitude.

`fwidth`
&ensp;   Camera sensor witdh in metres. This example is for a Nikkon D800 with full width sensor at ~36 x 24 mm, exactly 0.0359 m.

`fheight`
&ensp;   Camera sensor height in metres. This example is for a Nikkon D800 with full width sensor at ~36 x 24 mm, exactly 0.024 m.

`focallength`
&ensp;   Camera lens focal lentgh in metres. Compact cameras and zoon lenses can have a very different nominal and actual value, they can differ up to a 10%.

`roll`
&ensp;   Camera roll in degrees. If the camera is hand held, it is very difficult to keep it perfectly horizontal. The roll helps correcting lateral inclination of the camera. A negative roll will rotate the image to the right.

## Usage

Some examples are provided in the attached zip files with all the necessary files to run them.


## References

- Corripio, J. G. (2004) Snow surface albedo estimation using terrestrial photography, International Journal of Remote Sensing. 25(24), 5705–5729. [preprint](https://www.arolla.ethz.ch/georef/albedo.pdf)


- Some of the applications are described in these papers https://scholar.google.com/scholar?cites=4579335701175756671&as_sdt=2005&sciodt=0,5&hl=en


## Requirements

- python 3x
- argparse
- copy
- cv2
- json
- matplotlib
- numpy
- os
- osgeo
- sys
- time
- tkinter
- urllib.request
- yaml


## Installation

Just run the script with all the required python modules installed.

`python3 photogeoref.py` 
