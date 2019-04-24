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
from collections import defaultdict
from sklearn.externals import joblib

# use limits and offsets so that i can finish, also fix so that you dont query a million times for admit time
# implement a program wide limit on patients

class Application(object):
    def __init__(self):
        self.program_args = Parameters.parse_parameters()
        data_dir = self.program_args["dataset"]
        self.dataset = DataSet(data_dir=data_dir)
        self.data_access = DataAccess(data_dir=data_dir)
        # limit = self.program_args["limit"]
        # self.training_set, self.validation_set, self.input_shape, self.output_dim = self.get_data()

    def get_data(self):
        time_frames = self.dataset.get_time_frame_per_patient()

        if self.program_args["rrates" is True]:
            all_respiratory_rates = self.data_access.get_rrates()
            dict_with_rr_data = self.dataset.get_rr_data(time_frames, all_respiratory_rates)
        if self.program_args["alcohol" is True]:
            patients_with_alcohol_history = self.data_access.get_patients_with_alcohol_abuse()
            dict_of_patients_alcohol_abuse = self.dataset.alcohol_abuse_binary_dictionary(patients_with_alcohol_history, dict_with_rr_data)
        if self.program_args["potassium" is True]:
            potassium_rates = self.data_access.get_potassium()
            dict_with_median_potassium_rates = self.dataset.get_potassium_data(potassium_rates, time_frames)
        if self.program_args["sodium" is True]:
            sodium_rates = self.data_access.get_sodium()
            dict_with_median_sodium_rates = self.dataset.get_sodium_data(sodium_rates, time_frames)
        all_patients = self.dataset.make_all_patients()
        return all_patients

    def run(self):
        validation_set_fraction = float(self.program_args["validation_set_fraction"])
        test_set_fraction = float(self.program_args["test_set_fraction"])

        patient_ids_with_arrhythmias = self.data_access.get_patients_with_arrhythmias()

        rr = self.dataset.get_rr_data()
        y_with_patient_id = self.dataset.get_y(self.data_access, rr)
        y_true = self.dataset.delete_patient_ids(y_with_patient_id)  # this function returns np.array
        x = self.dataset.delete_patient_ids(rr)

        # later should be x = dataset.delete_patient_ids(features_of_all_patients)

        x_train, y_train, x_val, y_val, x_test, y_test = \
                self.dataset.split(x, y_true,
                                   validation_set_size=int(np.rint(validation_set_fraction*len(x))),
                                   test_set_size=int(np.rint(test_set_fraction*len(x))))

        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        y_pred = rf.predict_proba(x_test)
        feature_importance = rf.feature_importances_
        # save order so that we know which is which

        # saving output and model
        joblib.dump(rf, self.program_args["model_file"])  # want directory, not one file to be written over
        np.savez(self.program_args["importances_file"], feature_importance)
        np.savez(self.program_args["predictions_file"], y_pred)

        # Compare y_test and y_pred
        from sklearn.metrics import roc_auc_score

        y_pred = np.argmax(y_pred, axis=-1)

        auc_score = roc_auc_score(y_test, y_pred)
        with open(self.program_args["results_file"], "w") as results_file:
            print("AUC Score is", auc_score, file=results_file)

        # sklearn.metrices.roc_auc_score
        # sklearn.metrices.f1_score
        # also average_precision_score
        # sensitivity or specificity must be also calculated (usually choose the one closest to top left (1,1) of
        # roc curve, choose this threshold of specificity and sensitivity)
        # y_score is predicted
        # read what different metrices do and try multiple
        # TODO: Program argument to switch between test set and validation set here.
        # TODO: filter children?
        # https://github.com/MIT-LCP/mimic-code/blob/ddd4557423c6b0505be9b53d230863ef1ea78120/concepts/cookbook/potassium.sql
        # contains filtering for adults

        # TODO: Step 3 - save the model to an output directory and write the results to a file

        return


if __name__ == "__main__":
    app = Application()
    app.run()
