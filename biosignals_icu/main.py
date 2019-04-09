from biosignals_icu.dataset import DataSet
from biosignals_icu.data_access import DataAccess
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Application(object):
    def run(self):
        dataset = DataSet(data_dir="/cluster/work/karlen/data/mimic3")
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")

        patients = data_access.get_patients()
        zeros = np.zeros(len(patients))
        all_patients = dict(zip(patients, zeros))  # how do i get patient id out of array format?

        l = data_access.get_patients_with_astemizole()
        rr = dataset.get_rr_data()

        y = dataset.get_y(data_access)
        x = dataset.before_prediction_x(all_patients)
        y = dataset.before_prediction_y(y)

        validation_set_fraction = 0.1  # TODO: Make program argument
        test_set_fraction = 0.2  # TODO: Make program argument
        x_train, y_train, x_val, y_val, x_test, y_test = \
            dataset.split(x, y,
                          validation_set_size=int(np.rint(validation_set_fraction*len(all_patients))),
                          test_set_size=int(np.rint(test_set_fraction*len(all_patients))))

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
    app.run()
