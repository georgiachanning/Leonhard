"""
Copyright (C) 2019  ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import pandas as pd
import os.path
import numpy as np
import scipy.sparse
from dateutil import relativedelta
from dateutil import parser
from datetime import datetime
from biosignals_icu.data_access import DataAccess
import sqlite3
from unicodedata import normalize
from os.path import join


class DataSet(object):
    DB_FILE_NAME = "mimic3.db"

    def __init__(self, data_dir):
        self.db = self.connect(data_dir)

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataSet.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    def whatami(self, module):
        print(dir(module))

    def get_data(self):
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")
        # preprocess=data_access.{which_data}
        admit_time = data_access.get_admit_time(limit=100)
        end_window = []

        # date format= '2125-04-25 23:39:00'

        for x in range(0, admit_time.__len__()-1):
            date = admit_time[x][1]
            start_window = parser.parse(date, yearfirst=True, dayfirst=False)  # date must be python string, not unicode
            # print(start_window)
            # end_me = start_window + relativedelta(days=+1)
            # end_window.append((admit_time[x][0], end_me))

        feature_preprocess = data_access.get_rrates()
        feature_process = np.zeros_like(feature_preprocess)
        for x in range(0, feature_preprocess.__len__()-1):
            if feature_preprocess[x] <= end_window[x]:
                feature_process = feature_preprocess # delete events where time is less than window

        return feature_process

    def get_y(self):
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")
        x = data_access.get_patients()
        num_patients = data_access.get_num_rows('PATIENTS')
        patient_ids_with_arrhythmias = data_access.get_patients_with_arrhythmias()
        # y = np.zeros_like(x)
        y = [num_patients][2]
        for l in range(0, num_patients - 1):
            y[l][0] = x[l][0]
            if x[l][0] in patient_ids_with_arrhythmias:
                y[l][1] = 1

    def split(self, x, y, val_set_size, test_set_size):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_set_size, random_state=0)
        rest_index, test_index = next(sss.split(x, y))


        #TODO: Split off val from rest indices
        x_val, y_val = None, None
        x_test, y_test = x[test_index], y[test_index]
        x_train, y_train = None, None
        return x_train, y_train, x_val, y_val, x_test, y_test
