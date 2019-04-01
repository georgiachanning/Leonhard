from sklearn import svm
from sklearn import datasets
from biosignals_icu.data_access import DataAccess
from sklearn import tree
import graphviz

#https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql


class Predict(object):
    def __init__(self, data_dir):
        self.predict = Predict(data_dir=data_dir)

    def predict_me(self, x, y):
        data_access = DataAccess(data_dir="/cluster/work/karlen/data/mimic3")
        clf = svm.SVC(gamma=0.001, C=100.)

        X = [[0, 0], [1, 1]]
        Y = [0, 1]
        rf = tree.RandomForestClassifier()
        rf = rf.fit(X, Y)
        rf.predict_proba([[2., 2.]])

        dot_data = tree.export_graphviz(rf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("mimic")
