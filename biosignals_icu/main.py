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
from biosignals_icu.dataset import DataSet
from biosignals_icu.data_access import DataAccess
# from biosignals_icu.program_args import parse_parameters
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import defaultdict
from sklearn.externals import joblib

class Application(object):
    def run(self, data_dir, validation_set_fraction, test_set_fraction):
        dataset = DataSet(data_dir=data_dir)
        data_access = DataAccess(data_dir=data_dir)

        patient_ids_with_arrhythmias = data_access.get_patients_with_arrhythmias()

        rr = dataset.get_rr_data(data_access, limit=5000)
        y_with_patient_id = dataset.get_y(data_access, rr)
        y = dataset.delete_patient_ids(y_with_patient_id)  # this function returns np.array
        x = dataset.delete_patient_ids(rr)

        # later should be x = dataset.delete_patient_ids(features_of_all_patients)

        x_train, y_train, x_val, y_val, x_test, y_test = \
            dataset.split(x, y,
                          validation_set_size=int(np.rint(validation_set_fraction*len(x))),
                          test_set_size=int(np.rint(test_set_fraction*len(x))))

        x_train = np.array(x_train).reshape(-1, 1)
        x_test = np.array(x_test).reshape(-1, 1)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        y_pred = rf.predict_proba(x_test)
        feature_importance = rf.feature_importances_

        # saving output and model

        joblib.dump(rf, 'saved_model.pkl')  # want directory, not one file to be written over
        outfile = open('output.txt', 'a', 1)
        outfile.write(feature_importance)


        # Compare y_test and y_pred
        # TODO: Program argument to switch between test set and validation set here.
        # TODO: filter children?
        # https://github.com/MIT-LCP/mimic-code/blob/ddd4557423c6b0505be9b53d230863ef1ea78120/concepts/cookbook/potassium.sql
        # contains filtering for adults

        # TODO: Step 3 - save the model to an output directory and write the results to a file

        return


if __name__ == "__main__":
    app = Application()
    app.run("/cluster/work/karlen/data/mimic3", 0.1, 0.2)
