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
from dateutil import parser


class Application(object):
    def __init__(self):
        self.program_args = Parameters.parse_parameters()
        data_dir = self.program_args["dataset"]
        self.db = self.connect(data_dir)
        self.dataset = DataSet(data_dir)
        self.data_access = DataAccess(data_dir)
        self.loaded_patients = self.data_access.get_patients()

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, DataAccess.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    def other_data(self):
        a = self.data_access.get_patients_with_arrhythmias()
        b = self.data_access.get_patients_with_arrhythmiacs()
        difference = {}

        a_Without_b = []
        for patient in a:
            if patient in b.keys():
                continue
            else:
                a_Without_b.append(patient)

        for patient in b.values():
            patient_admit = list(self.db.execute("SELECT HADM_ID, ADMITTIME, SUBJECT_ID FROM ADMISSIONS "
                                                 "WHERE HADM_ID = '{patient_hadm}';".format(patient_hadm=patient[2])).fetchone())

            difference[patient[0]] = (parser.parse(patient[1]) - parser.parse(patient_admit[1])).total_seconds()
        avg_time_before_medication = sum(difference)/len(difference)
        # people were medicated on average within 11 hours 16 minutes and 17 seconds of admission,
        # so keep for 24/12 hours measurements, drop for smaller
        return avg_time_before_medication

    def get_data(self):
        time_frames = self.dataset.get_time_frame_per_patient()
        args_for_all_patients = {}

        if self.program_args["dyspnea"] is True:
            patients_with_dyspnea = self.data_access.get_patients_with_dyspnea()
            dict_with_dyspnea = self.dataset.binary_data_to_dict(patients_with_dyspnea, self.loaded_patients)
            args_for_all_patients["dyspnea"] = dict_with_dyspnea
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
        if self.program_args["epilepsy"] is True:
            patients_with_epilepsy = self.data_access.get_patients_with_epilepsy_history()
            dict_with_epilepsy = self.dataset.binary_data_to_dict(patients_with_epilepsy, self.loaded_patients)
            args_for_all_patients["epilepsy"] = dict_with_epilepsy
        if self.program_args["chest_pain"] is True:
            patients_with_chest_pain = self.data_access.get_patients_with_chest_pain()
            dict_with_chest_pain = self.dataset.binary_data_to_dict(patients_with_chest_pain, self.loaded_patients)
            args_for_all_patients["chest_pain"] = dict_with_chest_pain
        if self.program_args["heart_failure"] is True:
            patients_with_heart_failure = self.data_access.get_patients_with_heart_failure()
            dict_with_heart_failure = self.dataset.binary_data_to_dict(patients_with_heart_failure, self.loaded_patients)
            args_for_all_patients["heart_failure"] = dict_with_heart_failure
        if self.program_args["calcium"] is True:
            patients_with_calcium = self.data_access.get_calcium()
            dict_with_calcium = self.dataset.get_median_calcium(patients_with_calcium, time_frames)
            args_for_all_patients["calcium"] = dict_with_calcium
        if self.program_args["cocaine"] is True:
            patients_with_cocaine = self.data_access.get_patients_with_cocaine()
            dict_with_cocaine = self.dataset.binary_data_to_dict(patients_with_cocaine, self.loaded_patients)
            args_for_all_patients["cocaine"] = dict_with_cocaine
        if self.program_args["muscular_dystrophy"] is True:
            patients_with_md = self.data_access.get_patients_with_muscular_dystrophy()
            dict_with_md = self.dataset.binary_data_to_dict(patients_with_md, self.loaded_patients)
            args_for_all_patients["muscular_dystrophy"] = dict_with_md
        if self.program_args["cardiac_arrest"] is True:
            patients_with_cardiac_arrest = self.data_access.get_patients_with_cardiac_arrest()
            dict_with_ca = self.dataset.binary_data_to_dict(patients_with_cardiac_arrest, self.loaded_patients)
            args_for_all_patients["cardiac_arrest"] = dict_with_ca

        all_patients, order_of_labels = self.dataset.make_all_patients(**args_for_all_patients)
        return all_patients, order_of_labels

    def hyper_paramter(self, x_data, y_data):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score

        with open(self.program_args["hyperp_file"], "w") as hyper_parameter_file:
                hyper_parameter_file.write("Hyper Parameters File: " + '\n')

        c, r = np.asarray(y_data).shape
        y_data = np.asarray(y_data).reshape(c, )

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        scores = ['precision', 'recall']

        for score in scores:

            rf0 = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
            rf0.fit(x_data, y_data)
            best_params = rf0.best_params_
            means = rf0.cv_results_['mean_test_score']
            stds = rf0.cv_results_['std_test_score']

            with open(self.program_args["hyperp_file"], "a") as hyper_parameter_file:
                hyper_parameter_file.write("Best parameters set found on development set:" + '\n')
                for key, value in best_params.items():
                    hyper_parameter_file.write(key + ": ")
                    hyper_parameter_file.write(str(value) + '\n')
                hyper_parameter_file.write('\n' + "Grid scores on development set:")
                for mean, std, params in zip(means, stds, rf0.cv_results_['params']):
                    hyper_parameter_file.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params) + '\n')

            # num of trees
            n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
            for estimator in n_estimators:
                rf1 = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
                rf1.fit(x_data, y_data)
                train_pred = rf1.predict(x_data)
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_data, train_pred)
                auc_score = roc_auc_score(y_data, train_pred)
                with open(self.program_args["hyperp_file"], "a") as hyper_parameter_file:
                    hyper_parameter_file.write("Num " + str(estimator) + " estimators:" + '\n')
                    hyper_parameter_file.write("False positive rate: " + str(false_positive_rate) + '\n')
                    hyper_parameter_file.write("True positive rate: " + str(true_positive_rate) + '\n')
                    hyper_parameter_file.write("Thresholds: " + str(thresholds) + '\n')
                    hyper_parameter_file.write("AUC Score: " + str(auc_score) + '\n' + '\n')

            # max depth
            max_depths = np.linspace(1, 32, 32, endpoint=True)
            for max_depth in max_depths:
                rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
                rf.fit(x_data, y_data)
                train_pred = rf.predict(x_data)
                false_positive_rate, true_positive_rate, thresholds = roc_curve(y_data, train_pred)
                auc_score = roc_auc_score(y_data, train_pred)
                with open(self.program_args["hyperp_file"], "a") as hyper_parameter_file:
                    hyper_parameter_file.write("Num " + str(max_depth) + " as max depth:" + '\n')
                    hyper_parameter_file.write("False positive rate: " + str(false_positive_rate) + '\n')
                    hyper_parameter_file.write("True positive rate: " + str(true_positive_rate) + '\n')
                    hyper_parameter_file.write("Thresholds: " + str(thresholds) + '\n')
                    hyper_parameter_file.write("AUC Score: " + str(auc_score) + '\n' + '\n')

    def run(self):

        all_patients_features, order_of_labels = self.get_data()

        validation_set_fraction = float(self.program_args["validation_set_fraction"])
        test_set_fraction = float(self.program_args["test_set_fraction"])
        y_with_patient_id = self.dataset.get_y(self.data_access, all_patients_features)

        assert len(y_with_patient_id) == len(all_patients_features.keys())

        y_true = self.dataset.delete_patient_ids(y_with_patient_id)  # this function returns np.array
        x = self.dataset.delete_patient_ids(all_patients_features)

        x_train, y_train, x_val, y_val, x_test, y_test = \
            self.dataset.split(x, y_true,
                               validation_set_size=int(np.rint(validation_set_fraction*len(x))),
                               test_set_size=int(np.rint(test_set_fraction*len(x))))

        num_estimators = self.program_args["max_num_of_trees"]
        max_depth = self.program_args["max_depth_of_trees"]

        rf = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth)

        # which split of data
        if self.program_args["split"] == "train":
            rf.fit(x_train, y_train)
            x_data = x_train
            y_data = y_train

        if self.program_args["split"] == "validate":
            rf.fit(x_val, y_val)
            x_data = x_val
            y_data = y_val

        if self.program_args["split"] == "test":
            rf.fit(x_test, y_test)
            x_data = x_test
            y_data = y_test

        y_pred = rf.predict_proba(x_data)
        feature_importance = rf.feature_importances_

        # saving output and model
        joblib.dump(rf, self.program_args["model_file"])
        np.savez(self.program_args["importances_file"], feature_importance)
        np.savez(self.program_args["predictions_file"], y_pred)

        # Compare y_test and y_pred
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import average_precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import roc_curve

        y_pred = np.argmax(y_pred, axis=-1)
        auc_score = roc_auc_score(y_data, y_pred)
        f1_score = f1_score(y_data, y_pred)
        average_precision_score = average_precision_score(y_data, y_pred)
        recall_score = recall_score(y_data, y_pred)
        accuracy_score = accuracy_score(y_data, y_pred)
        hamming_loss = hamming_loss(y_data, y_pred)
        fpr, tpr, thresholds = roc_curve(y_data, y_pred)
        optimal_threshold_idx = np.argmin(np.linalg.norm(np.stack((fpr, tpr)).T -
                                                         np.repeat([[0., 1.]], fpr.shape[0], axis=0), axis=1))
        threshold = thresholds[optimal_threshold_idx]

        with open(self.program_args["results_file"], "w") as results_file:
            print("AUC Score is", auc_score, file=results_file)
            print("f1 Score is", f1_score, file=results_file)
            print("Average Precision Score is", average_precision_score, file=results_file)
            print("Recall Score is", recall_score, file=results_file)
            print("Accuracy Score is", accuracy_score, file=results_file)
            print("Hamming Loss is", hamming_loss, file=results_file)
            print("Specificity is", 1-fpr, file=results_file)
            print("Sensitivity is", tpr, file=results_file)
            print("Optimal Threshold:", threshold, file=results_file)
            print("Order of Labels: ", order_of_labels, file=results_file)
            print("Feature Importances: ", feature_importance, file=results_file)

        if self.program_args["split"] == "validate":
            self.hyper_paramter(x_data, y_data)

        return


if __name__ == "__main__":
    app = Application()
    app.run()
