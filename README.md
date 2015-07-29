# SpatialDM

QGIS Plugin to run data mining algorithms on spatial datasets.

## General Overview

SpatialDM is QGIS Plugin designed to run classification algorithms on spatial data. It is compatible with both multi-band raster layers and comma separated values (CSV) files. Currently three classifiers have been implemented:

* Decision Tree Classifier
* AdaBoost Classifier
* Random Forest Classifier

## Installation

Before installing the SpatialDM plugin ensure that you have QGIS, Python and Scikit-learn installed on your system (see the dependencies section).

To install the plugin just paste the SpatialDM directory in the following folders:

* UNIX/Mac: ***~/.qgis/python/plugins*** and ***(qgis_prefix)/share/qgis/python/plugins***
* Windows: ***~/.qgis/python/plugins*** and ***(qgis_prefix)/python/plugins***

Home directory (denoted by above ~) on Windows is usually something like *C:\Documents and Settings\(user)* (on Windows XP or earlier) or *C:\Users\(user)*.

**NOTE:-** By setting the *QGIS_PLUGINPATH* to an existing directory, you can add this path to the list of paths that are searched for plugins.

### Dependencies

* QGIS (version > 2.0)
* Python (>= 2.6 or >= 3.3) -> <https://www.python.org/downloads>
* NumPy (>= 1.6.1) -> <http://www.numpy.org>
* SciPy (>= 0.9) -> <http://www.scipy.org>
* Graphviz (optional to display DOT files) -> <http://www.graphviz.org>