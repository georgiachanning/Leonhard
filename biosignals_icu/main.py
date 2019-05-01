"""
Copyright (C) 2019 ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function
from biosignals_icu.dataset import DataSet
from biosignals_icu.data_access import DataAccess
from biosignals_icu.program_args import Parameters
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
import sqlite3
from os.path import join
# implement offset


class Application(object):
    def __init__(self):
        self.program_args = Parameters.parse_parameters()
        data_dir = self.program_args["dataset"]
        self.db = self.connect(data_dir)
        self.dataset = DataSet(data_dir)
        self.data_access = DataAccess(data_dir)

        self.loaded_patients = self.data_access.get_patients()
        self.all_patients_features = self.get_data()

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataAccess.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    def getsample(self):
        dict_1 = {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}
        dict_2 = {1: 1, 3: 9, 4: 16}
        dict_3 = {4: 8, 5: 10, 2: 4}
        dict_4 = {3: 0, 2: 0, 1: 0}
        dict_5 = {5: 4820230, 4: 239840, 3: 497340}
        args = {'a': dict_1, 'b': dict_2, 'c': dict_3, 'd': dict_4, 'e': dict_5}
        self.dataset.make_all_patients(**args)

    def get_data(self):
        time_frames = self.dataset.get_time_frame_per_patient()
        args_for_all_patients = {}

        if self.program_args["rrates"] is True:
            all_respiratory_rates = self.data_access.get_rrates()
            dict_with_rr_data = self.dataset.get_rr_data(time_frames, all_respiratory_rates)
            args_for_all_patients["rrates"] = dict_with_rr_data
        if self.program_args["alcohol"] is True:
            patients_with_alcohol_history = self.data_access.get_patients_with_alcohol_abuse()
            dict_of_patients_alcohol_abuse = self.dataset.binary_data_to_dict(patients_with_alcohol_history,
                                                                              self.loaded_patients)
            args_for_all_patients["alcohol"] = dict_of_patients_alcohol_abuse
        if self.program_args["potassium"] is True:
            potassium_rates = self.data_access.get_potassium()
            dict_with_median_potassium_rates = self.dataset.get_potassium_data(potassium_rates, time_frames)
            args_for_all_patients["potassium"] = dict_with_median_potassium_rates
        if self.program_args["sodium"] is True:
            sodium_rates = self.data_access.get_sodium()
            dict_with_median_sodium_rates = self.dataset.get_sodium_data(sodium_rates, time_frames)
            args_for_all_patients["sodium"] = dict_with_median_sodium_rates
        if self.program_args["blood_pressure"] is True:
            mean_blood_pressure_rates = self.data_access.get_mean_blood_pressure()
            dict_with_blood_pressure = self.dataset.get_median_blood_pressure(mean_blood_pressure_rates, time_frames)
            args_for_all_patients["blood_pressure"] = dict_with_blood_pressure
        if self.program_args["quinine"] is True:
            patients_with_quinine = self.data_access.get_patients_with_quinine()
            dict_with_quinine = self.dataset.binary_data_to_dict(patients_with_quinine, self.loaded_patients)
            args_for_all_patients["quinine"] = dict_with_quinine
        if self.program_args["astemizole"] is True:
            patients_with_astemizole = self.data_access.get_patients_with_astemizole()
            dict_with_astemizole = self.dataset.binary_data_to_dict(patients_with_astemizole, self.loaded_patients)
            args_for_all_patients["astemizole"] = dict_with_astemizole
        if self.program_args["terfenadine"] is True:
            patients_with_terfenadine = self.data_access.get_patients_with_terfenadine()
            dict_with_terfenadine = self.dataset.binary_data_to_dict(patients_with_terfenadine, self.loaded_patients)
            args_for_all_patients["terfenadine"] = dict_with_terfenadine
        if self.program_args["pulmonary_circulation_disorder"] is True:
            patients_with_pcd = self.data_access.get_patients_with_pulmonary_circulation_disorder()
            dict_with_pcd = self.dataset.binary_data_to_dict(patients_with_pcd, self.loaded_patients)
            args_for_all_patients["pcd"] = dict_with_pcd
        if self.program_args["lung_disease"] is True:
            patients_with_lung_disease = self.data_access.get_patients_with_lung_disease()
            dict_with_lung_disease = self.dataset.binary_data_to_dict(patients_with_lung_disease, self.loaded_patients)
            args_for_all_patients["lung_disease"] = dict_with_lung_disease
        if self.program_args["renal_failure"] is True:
            patients_with_renal_failure = self.data_access.get_patients_with_renal_failure()
            dict_with_renal_failure = self.dataset.binary_data_to_dict(patients_with_renal_failure,
                                                                       self.loaded_patients)
            args_for_all_patients["renal_failure"] = dict_with_renal_failure

        all_patients, order_of_labels = self.dataset.make_all_patients(**args_for_all_patients)
        np.savez(self.program_args["order_of_features_file"], order_of_labels)
        return all_patients

    def run(self):
        validation_set_fraction = float(self.program_args["validation_set_fraction"])
        test_set_fraction = float(self.program_args["test_set_fraction"])
        y_with_patient_id = self.dataset.get_y(self.data_access, self.all_patients_features)

        assert len(y_with_patient_id) == len(self.all_patients_features.keys())

        y_true = self.dataset.delete_patient_ids(y_with_patient_id)  # this function returns np.array
        x = self.dataset.delete_patient_ids(self.all_patients_features)

        x_train, y_train, x_val, y_val, x_test, y_test = \
            self.dataset.split(x, y_true,
                               validation_set_size=int(np.rint(validation_set_fraction*len(x))),
                               test_set_size=int(np.rint(test_set_fraction*len(x))))

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        y_pred = rf.predict_proba(x_test)
        feature_importance = rf.feature_importances_

        # saving output and model
        joblib.dump(rf, self.program_args["model_file"])
        np.savez(self.program_args["importances_file"], feature_importance)
        np.savez(self.program_args["predictions_file"], y_pred)


        # Compare y_test and y_pred
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import average_precision_score

        # y_pred = np.argmax(y_pred, axis=-1)

        # auc_score = roc_auc_score(y_true, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        f1_score = f1_score(y_true, y_pred)
        average_precision_score = average_precision_score(y_true, y_pred)

        with open(self.program_args["results_file"], "w") as results_file:
            print("AUC Score is", auc_score, file=results_file)
            print("f1 Score is", f1_score, file=results_file)
            print("Average Precision Score is", average_precision_score, file=results_file)

        # sklearn.metrices.roc_auc_score: done
        # sklearn.metrices.f1_score: done
        # also average_precision_score: done
        # sensitivity or specificity must be also calculated (usually choose the one closest to top left (1,1) of
        # roc curve, choose this threshold of specificity and sensitivity)
        # y_score is predicted
        # read what different metrices do and try multiple
        # TODO: Program argument to switch between test set and validation set here.
        # TODO: filter children?

        return


if __name__ == "__main__":
    app = Application()
    app.run()
