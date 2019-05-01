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
from argparse import ArgumentParser, Action


class Parameters(Action):

    @staticmethod
    def parse_parameters():
        # required = True, then if no arguments then throws error and shows help
        parser = ArgumentParser(description='mimic3 to cardiac arrhythmia prediction')
        parser.add_argument("--dataset", default="/cluster/work/karlen/data/mimic3",
                            help="The data set to be loaded from mimic3).")
        parser.add_argument("--validation_set_fraction", default=.1,
                            help="the fraction of the data that should be used for validation")
        parser.add_argument("--test_set_fraction", default=.2,
                            help="the fraction of the data that should be used for testing")
        parser.add_argument("--dataset_type", default="train",
                            help="Would you like to use the test, validation or train set?")
        parser.add_argument("--importances_file", default="/cluster/work/karlen/georgiachanning/importances.npz",
                            help="where should the output go?")
        parser.add_argument("--model_file", default="/cluster/work/karlen/georgiachanning/model.pkl",
                            help="where should the output go?")
        parser.add_argument("--results_file", default="/cluster/work/karlen/georgiachanning/results.txt",
                            help="where should the output go?")
        parser.add_argument("--predictions_file", default="/cluster/work/karlen/georgiachanning/predictions.npz",
                            help="where should the output go?")
        parser.add_argument("--max_depth_of_trees", default=5,
                            help="max depth of each tree")
        parser.add_argument("--max_num_of_trees", default=50,
                            help="max num of trees in forest")
        parser.add_argument("--num_features", default=2,
                            help="num of features passed")
        parser.add_argument("--get_kids", default=False,
                            help="should this training also include child patients?")
        parser.add_argument("--num_patients_to_load", default=20,
                            help="how many patients' data should be loaded?")
        parser.add_argument("--offset", default=300,
                            help="offset for number of patients to load")

        # following are all biosignals
        parser.add_argument("--heart_attacks", default=False,
                            help="include heart attack as feature")
        parser.add_argument("--rrates", default=False,
                            help="include respiratory rates?")
        parser.add_argument("--cocaine", default=False,
                            help="include cocaine history?")
        parser.add_argument("--quinine", default=False,
                            help="include whether patient is prescribed quinine?")
        parser.add_argument("--astemizole", default=False,
                            help="include whether patient is prescribed astemizole?")
        parser.add_argument("--terfenadine", default=False,
                            help="include whether patient is prescribed terfenadine?")
        parser.add_argument("--pulmonary_circulation_disorder", default=False,
                            help="include whether patient is diagnosed with lung disease?")
        parser.add_argument("--lung_disease", default=True,
                            help="include whether patient is diagnosed with pulmonary circulation disorder?")
        parser.add_argument("--renal_failure", default=False,
                            help="include whether patient is diagnosed with renal failure?")
        parser.add_argument("--heart_failure", default=False,
                            help="include whether patient is diagnosed with heart_failure")
        parser.add_argument("--muscular_dystrophy", default=False,
                            help="include whether patient is diagnosed with muscular dystrophy?")
        parser.add_argument("--alcohol", default=False,
                            help="include whether patient has history of alcohol abuse?")
        parser.add_argument("--epilespy", default=False,
                            help="include whether patient has history of epilespy?")
        parser.add_argument("--orthopnea", default=False,
                            help="include whether patient is diagnosed with orthopnea?")
        parser.add_argument("--chest_pain", default=False,
                            help="include whether patient has chest pain?")
        parser.add_argument("--cardiac_arrest", default=False,
                            help="include whether patient has cardiac arrest?")
        parser.add_argument("--dyspnea", default=False,
                            help="include whether patient has dyspnea?")
        parser.add_argument("--potassium", default=False,
                            help="include patient median potassium rates?")
        parser.add_argument("--sodium", default=False,
                            help="include patient median sodium rates?")
        parser.add_argument("--blood_pressure", default=True,
                            help="include patient blood pressure?")
        return vars(parser.parse_args())

