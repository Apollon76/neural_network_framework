from sklearn.model_selection import train_test_split
import pandas
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Learn neural network.')
parser.add_argument('-p', '--path', type=str, help='Path to train.csv')
parser.add_argument('-d', '--dest', type=str, help='Destination folder with prepared data')
args = parser.parse_args()

train = pandas.read_csv(args.path)
train, test = train_test_split(train, test_size=0.2)

os.mkdir(args.dest)
train.to_csv(os.path.join(args.dest, "train.csv"))
test.to_csv(os.path.join(args.dest, "test.csv"))

