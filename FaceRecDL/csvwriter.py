import csv
import pandas as pd


def create(filename, col_name):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(col_name)


def write(filename, row):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def read(filename):
    return pd.read_csv(filename)
