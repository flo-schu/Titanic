import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.linear_model as sklm

train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

# correct NaN values -> set to 0
# train.fillna("0")
#
# train.Age = [ if np.isnan(i) else i for i in train.Age]
# train.Embarked = [i if isinstance(i, str) else "0" for i in train.Embarked]
# train.Cabin = [i if isinstance(i, str) else "0" for i in train.Cabin]

# extract title and name from passenger list
lname = [i.split(", ")[0] for i in train.Name ]
rname = [i.split(", ")[1] for i in train.Name ]
title = [i.split(" ", maxsplit = 1)[0] for i in rname]
fname = [i.split(" ", maxsplit = 1)[1] for i in rname]
gender = [1 if i == "female" else 0 for i in train.Sex]
train = train.assign(lname = lname, fname = fname, title = title, gender = gender)

train.title = train["title"].astype("category")
train.Sex = train["Sex"].astype("category")
train.Cabin = train["Cabin"].astype("category")
train.Embarked = train["Embarked"].astype("category")
train.Ticket = train["Ticket"].astype("category")
train.lname = train["lname"].astype("category")

train.info() # get infos about the quality of the data
train.describe()
train.describe(include = 'category')

train.Age.isna()

import scipy.stats as stats

stats.pearsonr(train.Age, train.Survived)


plt.plot(x = np.array(train.Age), y = np.array(train.Survived))

type(train.Age)
