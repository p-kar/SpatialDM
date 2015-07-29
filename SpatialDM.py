# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SpatialDM
                                 A QGIS plugin
 Data Mining Algorithms on Spatial Data
                              -------------------
        begin                : 2015-06-24
        git sha              : $Format:%H$
        copyright            : (C) 2015 by pkar, mandark
        email                : pratyush.kar@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4.QtGui import *
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication, QObject, SIGNAL, QPoint, QVariant
from PyQt4.QtGui import QAction, QIcon, QMessageBox
# Initialize Qt resources from file resources.py
import resources_rc
# Import the code for the dialog
from SpatialDM_dialog import SpatialDMDialog
import subprocess
import os.path
import csv
import random
no_dep_issues = True
try:
    import sklearn
    from sklearn import tree, ensemble
    from sklearn.metrics import confusion_matrix
    from sklearn.externals.six import StringIO
    import pydot
    import numpy
except ImportError:
    str_warn = []
    str_warn.append('Scikit-learn not installed!')
    str_warn.append('Scikit-learn requries:')
    str_warn.append('\t- Python(>= 2.6 or >= 3.3)')
    str_warn.append('\t- NumPy (>= 1.6.1)')
    str_warn.append('\t- SciPy (>= 0.9)')
    str_warn.append('\nVisit http://scikit-learn.org/stable/install.html for detailed install instructions.')
    no_dep_issues = False
from qgis.core import QgsPoint, QgsRaster, QgsField
from qgis.utils import iface 


class SpatialDM:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'SpatialDM_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)

        # Create the dialog (after translation) and keep reference
        self.dlg = SpatialDMDialog()

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&SpatialDM')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'SpatialDM')
        self.toolbar.setObjectName(u'SpatialDM')

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('SpatialDM', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToRasterMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/SpatialDM/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Run Data Mining Algorithms on Spatial Data'),
            callback=self.run,
            parent=self.iface.mainWindow())
        self.initGUI()
        self.defineSignals()


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginRasterMenu(
                self.tr(u'&SpatialDM'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def initGUI(self):
        self.dlg.raster_combo_box.clear()
        self.dlg.bands_combo_box.clear()
        self.dlg.csv_dir_text_edit.clear()
        self.dlg.csv_checkBox.setEnabled(True)
        self.dlg.csv_checkBox.setChecked(False)
        self.initDTreeGUI()
        self.initABoostGUI()
        self.initRForestGUI()
        self.chooseSplit()

    def initDTreeGUI(self):
        self.dlg.criterion_combo_box.clear()
        self.dlg.splitter_combo_box.clear()
        self.criterion_list = ['gini', 'entropy']
        self.splitter_list = ['best', 'random']
        self.dlg.criterion_combo_box.addItems(self.criterion_list)
        self.dlg.splitter_combo_box.addItems(self.splitter_list)

    def initABoostGUI(self):
        self.dlg.algorithm_combo_box.clear()
        self.algorithm_list = ['SAMME.R', 'SAMME']
        self.dlg.algorithm_combo_box.addItems(self.algorithm_list)

    def initRForestGUI(self):
        self.dlg.criterion_combo_box_3.clear()
        self.dlg.splitter_combo_box_3.clear()
        self.criterion_list = ['gini', 'entropy']
        self.splitter_list = ['best', 'random']
        self.dlg.criterion_combo_box_3.addItems(self.criterion_list)
        self.dlg.splitter_combo_box_3.addItems(self.splitter_list)

    def chooseInputType(self):
        csv_state = self.dlg.csv_checkBox.isChecked()
        if(csv_state):
            self.dlg.raster_combo_box.setEnabled(False)
            self.dlg.csv_dir_text_edit.setEnabled(True)
            self.dlg.csv_dir_browse_button.setEnabled(True)
        else:
            self.dlg.raster_combo_box.setEnabled(True)
            self.dlg.csv_dir_text_edit.setEnabled(False)
            self.dlg.csv_dir_browse_button.setEnabled(False)
        self.updateBands()

    def updateBands(self):
        self.dlg.bands_combo_box.clear()
        csv_state = self.dlg.csv_checkBox.isChecked()
        if(csv_state):
            if(self.dlg.csv_dir_text_edit.toPlainText() == ""):
                return
            r = csv.reader(open(self.dlg.csv_dir_text_edit.toPlainText()))
            self.csv_band_list = r.next()
            self.bandCount = len(self.csv_band_list)
            self.dlg.bands_combo_box.addItems(self.csv_band_list)
        else:
            currIndex = self.dlg.raster_combo_box.currentIndex()
            if(currIndex == -1):
                return
            raster_layer = self.layer_list[currIndex]
            self.bandCount = raster_layer.bandCount()
            band_name_list = []
            for i in range(0,self.bandCount):
                band_name_list.append(raster_layer.bandName(i + 1))
            self.dlg.bands_combo_box.addItems(band_name_list)

    def chooseSplit(self):
        split_state = self.dlg.partition_checkBox.isChecked()
        if(split_state):
            self.dlg.split_perc_spinBox.setEnabled(True)
            self.dlg.disp_accu_checkBox.setEnabled(True)
            self.dlg.label_12.setEnabled(True)
        else:
            self.dlg.split_perc_spinBox.setEnabled(False)
            self.dlg.disp_accu_checkBox.setEnabled(False)
            self.dlg.label_12.setEnabled(False)

    def defineSignals(self):
        QObject.connect(self.dlg.raster_combo_box, SIGNAL("currentIndexChanged ( int )"), self.updateBands)
        QObject.connect(self.dlg.csv_dir_text_edit, SIGNAL("textChanged ()"), self.updateBands)
        QObject.connect(self.dlg.csv_checkBox, SIGNAL("stateChanged ( int )"), self.chooseInputType)
        QObject.connect(self.dlg.csv_dir_browse_button, SIGNAL("clicked ( bool )"), self.browseFile)
        QObject.connect(self.dlg.train_pushButton, SIGNAL("clicked ( bool )"), self.trainDecisionTree)
        QObject.connect(self.dlg.train_pushButton_2, SIGNAL("clicked ( bool )"), self.trainAdaBoost)
        QObject.connect(self.dlg.train_pushButton_3, SIGNAL("clicked ( bool )"), self.trainRandomForest)
        QObject.connect(self.dlg.partition_checkBox, SIGNAL("stateChanged ( int )"), self.chooseSplit)

    def browseFile(self):
        fname = QFileDialog.getOpenFileName(None, "Select CSV File", os.path.join(self.plugin_dir, 'test_datasets'), "*.csv *.txt")
        self.dlg.csv_dir_text_edit.setText(fname)

    def getTrainingDatafromCSV(self):
        f = open(self.dlg.csv_dir_text_edit.toPlainText())
        r = csv.reader(f)
        target_band = self.dlg.bands_combo_box.currentIndex()
        data_cols = []
        for i in range(0,self.bandCount):
            if(i != target_band):
                data_cols.append(i)
        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []
        self.data = []
        self.target = []
        for row in r:
            self.train_data.append(list(row[i] for i in data_cols))
            self.train_target.append(row[target_band])
        self.train_data.pop(0)
        self.train_target.pop(0)
        self.data = self.train_data
        self.target = self.train_target
        split_state = self.dlg.partition_checkBox.isChecked()
        if(split_state):
            test_data_cnt = round(len(self.train_data)*(1 - self.dlg.split_perc_spinBox.value()/100.0))
            while(len(self.test_data) < test_data_cnt):
                idx = random.randint(0, len(self.train_data) - 1)
                self.test_data.append(self.train_data[idx])
                self.test_target.append(self.train_target[idx])
                self.train_data.pop(idx)
                self.train_target.pop(idx)

    def getTrainingDatafromRaster(self):
        currIndex = self.dlg.raster_combo_box.currentIndex()
        raster_layer = self.layer_list[currIndex]
        target_band = self.dlg.bands_combo_box.currentIndex() + 1
        w = raster_layer.width()
        h = raster_layer.height()
        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []
        self.data = []
        self.target = []
        for i in range(0,w):
            for j in range(0,h):
                instance = []
                for k in range(1,self.bandCount + 1):
                    pos = QgsPoint(i, -j)
                    ident = raster_layer.dataProvider().identify(pos, QgsRaster.IdentifyFormatValue)
                    if(k == target_band):
                        self.train_target.append(ident.results().get(k))
                    else:
                        instance.append(ident.results().get(k))
                self.train_data.append(instance)
        self.data = self.train_data
        self.target = self.train_target
        split_state = self.dlg.partition_checkBox.isChecked()
        if(split_state):
            test_data_cnt = round(len(self.train_data)*(1 - self.dlg.split_perc_spinBox.value()/100.0))
            while(len(self.test_data) < test_data_cnt):
                idx = random.randint(0, len(self.train_data) - 1)
                self.test_data.append(self.train_data[idx])
                self.test_target.append(self.train_target[idx])
                self.train_data.pop(idx)
                self.train_target.pop(idx)

    def run(self):
        """Run method that performs all the real work"""
        if(no_dep_issues == False):
            QMessageBox.warning(self.dlg, "Dependency Issues", "%s" % ('\n'.join(str_warn)))
            return
        self.initGUI()
        layers = self.iface.legendInterface().layers()
        self.layer_list = []
        self.layer_name_list = []
        for layer in layers:
            if(str(type(layer)) == "<class 'qgis._core.QgsRasterLayer'>"):
                self.layer_list.append(layer)
                self.layer_name_list.append(layer.name())
        self.dlg.raster_combo_box.addItems(self.layer_name_list)
        if(len(self.layer_list) == 0):
            self.dlg.csv_checkBox.setChecked(True)
            self.dlg.csv_checkBox.setEnabled(False)
        self.chooseInputType()
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

    def showConfusionMatrix(self, clf):
        self.predict = clf.predict(self.data)
        conf_matrix = confusion_matrix(self.target, self.predict)
        s = [[str(e) for e in row] for row in conf_matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        QMessageBox.information(self.dlg, "Confusion Matrix", "%s" % ('\n'.join(table)))
        # print '\n'.join(table)
        # print conf_matrix

    def trainDecisionTree(self):
        csv_state = self.dlg.csv_checkBox.isChecked()
        if(csv_state):
            self.getTrainingDatafromCSV()
        else:
            self.getTrainingDatafromRaster()
        idx = self.dlg.criterion_combo_box.currentIndex()
        self.criterion = self.criterion_list[idx]
        idx = self.dlg.splitter_combo_box.currentIndex()
        self.splitter = self.splitter_list[idx]
        self.max_depth = None if (self.dlg.max_depth_spinBox.value() == 0) else self.dlg.max_depth_spinBox.value()
        self.min_samples_split = self.dlg.min_samples_split_spinBox.value()
        self.min_samples_leaf = self.dlg.min_samples_leaf_spinBox.value()
        self.min_weight_fraction_leaf = self.dlg.min_weight_fraction_leaf_spinBox.value()
        self.max_leaf_nodes = None if (self.dlg.max_leaf_nodes_spinBox.value() == 0) else self.dlg.max_leaf_nodes_spinBox.value()
        
        clf = tree.DecisionTreeClassifier(\
            criterion = self.criterion, \
            splitter = self.splitter, \
            max_depth = self.max_depth, \
            min_samples_split = self.min_samples_split, \
            min_samples_leaf = self.min_samples_leaf, \
            min_weight_fraction_leaf = self.min_weight_fraction_leaf, \
            max_leaf_nodes = self.max_leaf_nodes)
        clf = clf.fit(self.train_data, self.train_target)
        split_state = self.dlg.partition_checkBox.isChecked()
        disp_acc = self.dlg.disp_accu_checkBox.isChecked()
        if(split_state and disp_acc):
            train_acc = clf.score(self.train_data, self.train_target)*100
            test_acc = clf.score(self.test_data, self.test_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Training Data Accuracy: %.2f%c\nTest Data Accuracy: %.2f%c" % (train_acc, '%', test_acc, '%'))
        else:
            train_acc = clf.score(self.train_data, self.train_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Accuracy: %.2f%%" % (train_acc))
        checkstate = self.dlg.conf_matrix_combo_box.isChecked()
        if(checkstate):
            self.showConfusionMatrix(clf)
        output_file_path = os.path.join(self.plugin_dir, 'tree.dot')
        tree.export_graphviz(clf, out_file = output_file_path)
        # graph = pydot.graph_from_dot_file(output_file_path)
        # graph.write_png(os.path.join(self.plugin_dir, 'output.png'))
        # os.system('dot -Tpng ' + os.path.join(self.plugin_dir, 'tree.dot')\
             # + ' -o ' + os.path.join(self.plugin_dir, 'output.png'))
        # os.system('open ' + os.path.join(self.plugin_dir, 'tree.dot'))

    def trainAdaBoost(self):
        csv_state = self.dlg.csv_checkBox.isChecked()
        if(csv_state):
            self.getTrainingDatafromCSV()
        else:
            self.getTrainingDatafromRaster()
        idx = self.dlg.algorithm_combo_box.currentIndex()
        self.algorithm = self.algorithm_list[idx];
        self.n_estimators = self.dlg.n_estimators_spinBox.value()
        self.learning_rate = self.dlg.learning_rate_spinBox.value()
        # self.random_state = None if (self.dlg.random_state_spinBox.value() == 0) else self.dlg.random_state_spinBox.value()

        clf = ensemble.AdaBoostClassifier(\
            n_estimators = self.n_estimators, \
            learning_rate = self.learning_rate, \
            algorithm = self.algorithm)
            # random_state = self.random_state)
        clf = clf.fit(self.train_data, self.train_target)
        split_state = self.dlg.partition_checkBox.isChecked()
        disp_acc = self.dlg.disp_accu_checkBox.isChecked()
        if(split_state and disp_acc):
            train_acc = clf.score(self.train_data, self.train_target)*100
            test_acc = clf.score(self.test_data, self.test_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Training Data Accuracy: %.2f%c\nTest Data Accuracy: %.2f%c" % (train_acc, '%', test_acc, '%'))
        else:
            train_acc = clf.score(self.train_data, self.train_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Accuracy: %.2f%%" % (train_acc))
        checkstate = self.dlg.conf_matrix_combo_box.isChecked()
        if(checkstate):
            self.showConfusionMatrix(clf)

    def trainRandomForest(self):
        csv_state = self.dlg.csv_checkBox.isChecked()
        if(csv_state):
            self.getTrainingDatafromCSV()
        else:
            self.getTrainingDatafromRaster()
        idx = self.dlg.criterion_combo_box_3.currentIndex()
        self.criterion = self.criterion_list[idx]
        idx = self.dlg.splitter_combo_box_3.currentIndex()
        self.splitter = self.splitter_list[idx]
        self.max_depth = None if (self.dlg.max_depth_spinBox_3.value() == 0) else self.dlg.max_depth_spinBox_3.value()
        self.min_samples_split = self.dlg.min_samples_split_spinBox_3.value()
        self.min_samples_leaf = self.dlg.min_samples_leaf_spinBox_3.value()
        self.min_weight_fraction_leaf = self.dlg.min_weight_fraction_leaf_spinBox_3.value()
        self.max_leaf_nodes = None if (self.dlg.max_leaf_nodes_spinBox_3.value() == 0) else self.dlg.max_leaf_nodes_spinBox_3.value()
        self.bootstrap = self.dlg.bootstrap_checkBox.isChecked()
        self.oob_score = self.dlg.oob_score_checkBox.isChecked()
        self.warm_start = self.dlg.warm_start_checkBox.isChecked()
        self.n_jobs = self.dlg.n_jobs_spinBox.value()

        clf = ensemble.RandomForestClassifier(\
            criterion = self.criterion, \
            max_depth = self.max_depth, \
            min_samples_split = self.min_samples_split, \
            min_samples_leaf = self.min_samples_leaf, \
            min_weight_fraction_leaf = self.min_weight_fraction_leaf, \
            max_leaf_nodes = self.max_leaf_nodes, \
            bootstrap = self.bootstrap, \
            oob_score = self.oob_score, \
            n_jobs = self.n_jobs, \
            warm_start = self.warm_start)
        clf = clf.fit(self.train_data, self.train_target)
        split_state = self.dlg.partition_checkBox.isChecked()
        disp_acc = self.dlg.disp_accu_checkBox.isChecked()
        if(split_state and disp_acc):
            train_acc = clf.score(self.train_data, self.train_target)*100
            test_acc = clf.score(self.test_data, self.test_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Training Data Accuracy: %.2f%c\nTest Data Accuracy: %.2f%c" % (train_acc, '%', test_acc, '%'))
        else:
            train_acc = clf.score(self.train_data, self.train_target)*100
            QMessageBox.information(self.dlg, "Accuracy", "Accuracy: %.2f%%" % (train_acc))
        checkstate = self.dlg.conf_matrix_combo_box.isChecked()
        if(checkstate):
            self.showConfusionMatrix(clf)
