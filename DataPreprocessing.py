from sklearn.preprocessing import LabelEncoder

# Removing outliers

train = train[train.item_price<100000]
train = train[train.item_price>0]
train = train[train.item_cnt_day<999]

# Some shop names are duplicate

# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11


shops.loc[shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"','shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['shop_city'] = shops['shop_name'].map(lambda x: x.split(' ')[0])
shops['shop_category'] = shops['shop_name'].map(lambda x: x.split(' ')[1]).str.lower()
shops.loc[shops['shop_city'] == '!Якутск', 'shop_city'] = 'Якутск'
occurences = shops['shop_category'].value_counts()
shops['shop_category'] = shops['shop_category'].apply( lambda x: x if (occurences[x]>2) else "Other")
shops['shop_city_encoded'] = LabelEncoder().fit_transform(shops['shop_city'])
shops['shop_category_encoded'] = LabelEncoder().fit_transform(shops['shop_category'])


categories['category_type'] = categories['item_category_name'].map(lambda x: x.split(' ')[0])
occurences = categories['category_type'].value_counts()
categories['category_type'] = categories['category_type'].apply( lambda x: x if (occurences[x]>2) else "OtherCategory")
categories['category_type_encoded'] = LabelEncoder().fit_transform(categories['category_type'])

categories['category_sub-type'] = categories['item_category_name'].map(lambda x: x.split(' - ')[len(x.split(' - '))-1])
categories['category_sub-type_encoded'] = LabelEncoder().fit_transform(categories['category_sub-type'])



import re
def name_correction(x):
    x = x.lower() # all letters lower case
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters
    x = x.replace('  ', ' ') # replace double spaces with single spaces
    x = x.strip() # remove leading and trailing white space
    return x

# split item names by first bracket
items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

# replace special characters and turn to lower case
items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

# fill nulls with '0'
items = items.fillna('0')

items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

# return all characters except the last if name 2 is not "0" - the closing bracket
items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")

items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"

group_sum = items.groupby(["type"]).agg({"item_id": "count"})
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )
items = items.drop(["type"], axis = 1)

items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)