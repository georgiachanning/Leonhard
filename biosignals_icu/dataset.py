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
from dateutil import parser
from datetime import timedelta
from biosignals_icu.data_access import DataAccess
import sqlite3
from os.path import join
from statistics import median


class DataSet(object):
    DB_FILE_NAME = "mimic3.db"

    def __init__(self, data_dir):
        self.db = self.connect(data_dir)

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataSet.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    def get_rr_data(self):
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")
        # preprocess=data_access.{which_data}
        admit_time = data_access.get_admit_time()
        feature_processed = np.zeros((len(admit_time), 2))
        end_windows = []
        # date format= '2125-04-25 23:39:00'

        # all_patients[patient_id] = x, y

        for i in range(0, len(admit_time)):
            current_patient_id, date = admit_time[i]
            start_window = parser.parse(date)
            end_me = start_window + timedelta(days=1)
            end_windows.append((current_patient_id, end_me))

        feature_preprocess = data_access.get_rrates()
        for i in range(0, len(feature_preprocess[0])):
            if feature_preprocess[i][1] <= end_windows[i]:
                feature_preprocess.remove(feature_preprocess[i])

        # Find median
        per_patient = {}
        for i in range(len(feature_preprocess[0])):
            patient_id, feature_value = feature_preprocess[i][0], feature_preprocess[i][2]
            if patient_id in per_patient:
                per_patient[patient_id].append(feature_value)
            else:
                per_patient[patient_id] = [feature_value]

        for patient_id in per_patient.keys():
            per_patient[patient_id][1] = np.median(per_patient[patient_id][1])

        return per_patient

    def get_y(self, data_access):
        x = data_access.get_patients()
        patient_ids_with_arrhythmias = data_access.get_patients_with_arrhythmias()
        y = np.zeros((len(x), 2))
        for i in range(len(x)):
            y[i][0] = x[i][0]
            if x[i][0] in patient_ids_with_arrhythmias:
                y[i][1] = 1

        return y

    def before_prediction_x(self, x):
        np.delete(x, 0, 0)
        return x

    def before_prediction_y(self, y):
        np.delete(y, 0, 0)
        return y

    def split(self, x, y, validation_set_size, test_set_size):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_set_size, random_state=0)
        rest_index, test_index = next(sss.split(x, y))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_size, random_state=0)
        train_index, val_index = next(sss.split(x[rest_index], y[rest_index]))

        x_test, y_test = x[test_index], y[test_index]
        x_val, y_val = x[rest_index][val_index], y[rest_index][val_index]
        x_train, y_train = x[rest_index][train_index], y[rest_index][train_index]
        return x_train, y_train, x_val, y_val, x_test, y_test
