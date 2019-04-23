"""
Copyright (C) 2019  Georgia Channing, ETH Zurich

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
from numpy import append
from dateutil import parser
from datetime import timedelta
from biosignals_icu.data_access import DataAccess
import sqlite3
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

    def make_all_patients(self, data_access):
        all_patients = data_access.get_patients() # should be get adult patients\
        features_of_all_patients = {}
        for key in all_patients:
            features_of_all_patients.setdefault(key, []).append(8)
        return features_of_all_patients

    def get_rr_data(self, limit):
        admit_time = self.data_access.get_admit_time()
        end_windows = {}
        # date format= '2125-04-25 23:39:00'

        for (current_patient_id, date) in admit_time:
            start_window = parser.parse(date)
            end_me = start_window + timedelta(days=1)
            end_windows[current_patient_id] = end_me

        all_respiratory_rates = self.data_access.get_rrates(limit)

        # Find median
        per_patient = {}
        rr_by_patient_id = {}
        for (patient_id, measurement_time, feature_value) in all_respiratory_rates:
            measurement_time = parser.parse(measurement_time)
            max_time_allowed = end_windows[patient_id]
            if not measurement_time <= max_time_allowed:
                continue

            if isinstance(feature_value, float):  # Ensure there is a measurement number.
                if patient_id in per_patient:
                    per_patient[patient_id].append(feature_value)
                else:
                    per_patient[patient_id] = [feature_value]

        for patient_id in per_patient.keys():
            rr_by_patient_id[patient_id] = np.median(per_patient[patient_id])

        return rr_by_patient_id

    def get_y(self, data_access, x):
        # x = data_access.get_patients()
        patient_ids_with_arrhythmias = data_access.get_patients_with_arrhythmias()
        y = {}
        for key in x:
            if key in patient_ids_with_arrhythmias:
                y[key] = 1
            else:
                y[key] = 0
        return y

    def delete_patient_ids(self, data_set):
        processed_data_set = np.array(0)

        '''for value in data_set.values():  # this needs to be sorted!!!
            processed_data_set = append(processed_data_set, value)  # should be all except index 0'''

        for i in range(len(data_set)):
            key = sorted(data_set.keys())[i]
            value = data_set[key]
            processed_data_set = append(processed_data_set, value)

        return processed_data_set

    def split(self, x, y, validation_set_size, test_set_size):
        from operator import itemgetter
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_set_size, random_state=0)
        rest_index, test_index = next(sss.split(x, y))

        rest_x = [x[i] for i in rest_index]
        rest_y = [y[i] for i in rest_index]

        # something is not acceptable about rest_index formatting

        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_size, random_state=0)
        train_index, val_index = next(sss.split(rest_x, rest_y))

        x_test, y_test = [x[i] for i in test_index], [y[i] for i in test_index]
        x_val, y_val = [x[i] for i in val_index], [y[i] for i in val_index]
        x_train, y_train = [x[i] for i in train_index], [y[i] for i in train_index]

        '''x_val, y_val = x[rest_index][val_index], y[rest_index][val_index]
        x_train, y_train = x[rest_index][train_index], y[rest_index][train_index]'''
        return x_train, y_train, x_val, y_val, x_test, y_test