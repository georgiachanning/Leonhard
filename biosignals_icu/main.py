from biosignals_icu.dataset import DataSet
from biosignals_icu.data_access import DataAccess
from biosignals_icu.predict import Predict
import numpy as np


class Application(object):
    def run(self):
        dataset = DataSet(data_dir="/cluster/work/karlen/data/mimic3")
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")
        predict = Predict(data_dir="/cluster/work/karlen/data/mimic3")

        patients = data_access.get_patients()
        zeros = np.zeros(len(patients))
        all_patients = {}
        all_patients = dict(zip(patients, zeros)) # how do i get patient id out of array format?

        l = data_access.get_patients_with_astemizole()
        rr = dataset.get_rr_data()
        a = data_access.get_patients_with_lung_disease()
        y = dataset.get_y()
        x = dataset.before_prediction_x(all_patients)
        y = dataset.before_prediction_y(y)
        predict.predict_me(x, y)

        # new idea:  get all data the first time and store it in the all_patients thing,
        # then for each x, just pull out the things i want for that trial

        #dataset.split(x,y,...,...)

        # TODO:  icd codes needs to be set so that they CONTAIN the id_set{}
        # TODO: Median aggregate all values for time window for each patient.
        #       / Filter for patients with arrhythmia only those values T hours before diagnostic code
        #       / Filter for patients without arrhythmia only those values T hours before last RR measurement

        # TODO: Remove first columns (ID and timestamp) before predicting

        # TODO: filter children?
        # https://github.com/MIT-LCP/mimic-code/blob/ddd4557423c6b0505be9b53d230863ef1ea78120/concepts/cookbook/potassium.sql
        # contains filtering for adults



        #patients_with_arrhythmias_in_window[]

        #x_train, y_train, x_val, y_val, x_test, y_test = dataset.split(x, y, val_set_size=999, test_set_size=999)


        # TODO: Step 1 - train model on x_train, y_train
        #rf = RandomForestClassifier()
        #rf.fit(x, y)

        # TODO: Step 2 - evaluate trained model on x_test, y_test

        # TODO: Step 3 - save the model to an output directory and write the results to a file

        return


if __name__ == "__main__":
    app = Application()
    app.run()
