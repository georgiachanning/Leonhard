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

import os
from argparse import ArgumentParser, Action, ArgumentTypeError


class ReadableDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise ArgumentTypeError("readable_dir:{} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise ArgumentTypeError("readable_dir:{} is not a readable dir".format(prospective_dir))

    def parse_parameters(self):
        parser = ArgumentParser(description='mimic3 to cardiac arrhythmia prediction')
        parser.add_argument("--dataset", default="/cluster/work/karlen/data/mimic3",
                            help="The data set to be loaded from mimic3).")
        parser.add_argument("--dataset_type", default="train",
                            help="Would you like to use the test, validation or train set?")
        parser.add_argument("--output_file", default="output.txt",
                            help="where should the output go?")
        parser.add_argument("--max_depth_of_trees", default=3000,
                            help="max depth of each tree")
        parser.add_argument("--max_num_of_trees", default=3000,
                            help="max num of trees in forest")
        parser.add_argument("--num_features", default=1,
                            help="num of features passed")

        return vars(parser.parse_args())

