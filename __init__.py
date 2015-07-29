# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SpatialDM
                                 A QGIS plugin
 Data Mining Algorithms on Spatial Data
                             -------------------
        begin                : 2015-06-24
        copyright            : (C) 2015 by pkar, mandark
        email                : pratyush.kar@gmail.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load SpatialDM class from file SpatialDM.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .SpatialDM import SpatialDM
    return SpatialDM(iface)
