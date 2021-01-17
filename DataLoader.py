import pandas as pd
import numpy as np

categories = pd.read_csv("KaggleData/item_categories.csv")
items = pd.read_csv("KaggleData/items.csv")
train = pd.read_csv("KaggleData/sales_train.csv")
sample_sub = pd.read_csv("KaggleData/sample_submission.csv")
shops = pd.read_csv("KaggleData/shops.csv")
test = pd.read_csv("KaggleData/test.csv")

categories_translated = pd.read_csv("TranslatedData/item_categories-translated.csv")
items_translated = pd.read_csv("TranslatedData/items-translated.csv")
shops_translated = pd.read_csv("TranslatedData/shops-translated.csv")
