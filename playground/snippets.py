from foolbox.api import *


with open(data_path+"fed_funds.p", mode="rb") as fname:
    effr = pickle.load(fname)["effective"]

with open(data_path + "fed_funds_futures_settle.p", mode="rb") as fname:
    ff = pickle.load(fname)
