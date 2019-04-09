from sklearn import svm
from sklearn import datasets
from biosignals_icu.data_access import DataAccess
from sklearn import tree
import graphviz
import sqlite3
from os.path import join
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql


class Predict(object):
    DB_FILE_NAME = "mimic3.db"

    def __init__(self, data_dir):
        self.db = self.connect(data_dir=data_dir)

    def connect(self, data_dir):
        db = sqlite3.connect(join(data_dir, Predict.DB_FILE_NAME),
                             check_same_thread=False,
                             detect_types=sqlite3.PARSE_DECLTYPES)
        return db

    def predict_me(self, x, y):  # also parameters X, Y
        svm.SVC(gamma=0.001, C=100.)
        X = [[0, 0], [1, 1]]
        Y = [0, 1]
        rf = RandomForestClassifier()
        rf.fit(x, y)
        # rf.feature_importances_  gini
        rf.predict_proba([[2., 2.]])  # what are these input numbers??

        '''dot_data = tree.export_graphviz(rf, out_file='tree.dot')
        graph = graphviz.Source(dot_data)
        graph.render("mimic")'''

    def do_i_work(self):
        iris = load_iris()
        rf = RandomForestClassifier()
        rf = rf.fit(iris.data, iris.target)
        print(rf.feature_importances_)
        print(rf.predict([[0, 0, 0, 0]]))
        dot_data = tree.export_graphviz(rf, out_file='tree.dot')
        graph = graphviz.Source(dot_data)
        graph.render("mimic")