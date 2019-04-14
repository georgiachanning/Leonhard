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
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Application(object):
    def run(self, data_dir, validation_set_fraction, test_set_fraction):
        dataset = DataSet(data_dir=data_dir)
        data_access = DataAccess(data_dir=data_dir)

        rr = dataset.get_rr_data(data_access)
        # all_patients = dict(zip(patients_with_arrhythmias, rr))

        y_with_patient_id = dataset.get_y(data_access)
        y = dataset.before_training_y(y_with_patient_id)
        x = dataset.before_training_x(rr)

        print(len(x) == len(y))

        x_train, y_train, x_val, y_val, x_test, y_test = \
            dataset.split(x, y,
                          validation_set_size=int(np.rint(validation_set_fraction*len(x))),
                          test_set_size=int(np.rint(test_set_fraction*len(x))))

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        # TODO: Program argument to switch between test set and validation set here.
        y_pred = rf.predict_proba(x_test)

        # Compare y_test and y_pred

        # new idea:  get all data the first time and store it in the all_patients thing,
        # then for each x, just pull out the things i want for that trial

        #dataset.split(x,y,...,...)

        # TODO: filter children?
        # https://github.com/MIT-LCP/mimic-code/blob/ddd4557423c6b0505be9b53d230863ef1ea78120/concepts/cookbook/potassium.sql
        # contains filtering for adults

        # TODO: Step 3 - save the model to an output directory and write the results to a file

        return


if __name__ == "__main__":
    app = Application()
    app.run("/cluster/work/karlen/data/mimic3", 0.1, 0.2)
