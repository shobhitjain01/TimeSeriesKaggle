# Repeating FeatureGeneration notebook as a python script

from sklearn.preprocessing import LabelEncoder

train['item_cnt_day'] = train['item_cnt_day'].clip(0,20)

# Adding revenue now as it will be lost later when merged into monthly data
train['daily_revenue'] = train['item_price'] * train['item_cnt_day']

# Features to be extracted from date
train['date']=pd.to_datetime(train['date'],format="%d.%m.%Y")
train['month']=train['date'].dt.strftime("%m");
train['month']=train['month'].astype('int64');
train['year']=train['date'].dt.strftime("%Y");
train['year']=train['year'].astype('int64');
train['days_in_month'] = train['date'].dt.days_in_month

# Grouping train data into monthly data
monthly_data = train.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day':'sum','item_price':'mean','daily_revenue':'mean','month':'mean','year':'mean','days_in_month':'mean'}).reset_index(); # groupby automatically removes the column not mentioned, reset_index doesn't make the groupby columns as index
# monthly_data = monthly_data.rename(columns={'date_block_num':'month_block_num','item_price':'mean_item_price','item_cnt_day':'item_cnt_month','daily_revenue':'monthly_revenue'});
monthly_data['item_cnt_day'] = monthly_data['item_cnt_day'].clip(0,20)





## !!! Here we combine test data points to train
# We create test points in the train data set
date_block_df = pd.DataFrame(list(range(34)),columns=['date_block_num']);
date_block_df['month'] = date_block_df['date_block_num']%12+1
date_block_df['year'] = 2013+date_block_df['date_block_num']//12
month_days = pd.Series([0,31,28,31,30,31,30,31,31,30,31,30,31]);
date_block_df['days_in_month'] = date_block_df['month'].map(month_days)

date_block_df['key']=0;
test_copy = test.copy();
test_copy['key']=0;
test_copy = test_copy.merge(date_block_df);  # creates a cartesian product of all (shop, item) pairs in test will all date_block_num values
test_copy = test_copy.drop(['ID','key'],axis=1);

monthly_data = pd.merge(test_copy,monthly_data,on=['shop_id','item_id','date_block_num','month','year','days_in_month'],how='left') 


# This will shift average counts towards zero
monthly_data.fillna(0,inplace=True);



# Making test like train and adding -1 to the columns that will be deleted later
test = test.drop(['ID'],axis=1)
test['date_block_num'] = 34;
test['item_cnt_day'] = -1;
test['item_price'] = -1;
test['daily_revenue'] = -1;
test['month'] = 11
test['year'] = 2015
test['days_in_month'] = 31

monthly_data = pd.concat([monthly_data, test], ignore_index=True, sort=False)

# Text features

categories_translated['category_type'] = categories_translated['item_category_name_translated'].map(lambda x: x.split(' - ')[0])
categories_translated['category_sub-type'] = categories_translated['item_category_name_translated'].map(lambda x: x.split(' - ')[len(x.split(' - '))-1])
categories_translated['category_type_encoded'] = LabelEncoder().fit_transform(categories_translated['category_type'])
categories_translated['category_sub-type_encoded'] = LabelEncoder().fit_transform(categories_translated['category_sub-type'])
shops_translated['shop_city'] = shops_translated['shop_name_translated'].map(lambda x: x.split(' ')[0])
shops_translated['shop_city_encoded'] = LabelEncoder().fit_transform(shops_translated['shop_city'])

# Merging files
monthly_data = pd.merge(monthly_data,items,on=['item_id'],how='inner')
monthly_data = pd.merge(monthly_data,categories_translated,on=['item_category_id'],how='inner')
monthly_data = pd.merge(monthly_data,shops_translated,on=['shop_id'],how='inner')

# Calculating Lagged features
def calculate_lag(lag,monthly_data,col):
    l=monthly_data.copy()
    l['date_block_num'] += lag
    l = l[['shop_id','item_id','date_block_num',col]]
    l = l.rename(columns={col:col+'_lag'+str(lag)});
    monthly_data = pd.merge(monthly_data,l,on=['shop_id','item_id','date_block_num'],how='left')
    return monthly_data

# Calculating Mean encoded features
def create_mean_features(train,monthly_data,groupby_cols,col_to_avg,new_col_name):
    cur_group = train.groupby(groupby_cols).agg({col_to_avg:'mean'}).reset_index();
    cur_group = cur_group.rename(columns={col_to_avg:new_col_name});
    monthly_data = pd.merge(monthly_data,cur_group, on = groupby_cols,how = 'left');
    return monthly_data

# lag_window affects this
for i in ['item_cnt_day','item_price','daily_revenue']:
    for j in range(0,lag_window):
        monthly_data = calculate_lag(j+1,monthly_data,i)
        
# Adding data to train to be used in calculating mean encoded features later
train = pd.merge(train,items,on=['item_id'],how='inner')
train = pd.merge(train,categories_translated,on=['item_category_id'],how='inner')
train = pd.merge(train,shops_translated,on=['shop_id'],how='inner')

# Mean encoded features

# Average of overall items sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num'],'item_cnt_day','item_cnt_avg_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_month')
# Average of overall items per item sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','item_id'],'item_cnt_day','item_cnt_avg_item_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_item_month')
# Average of overall items per shop sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_id'],'item_cnt_day','item_cnt_avg_shop_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_shop_month')
# Average of overall items per category sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','item_category_id'],'item_cnt_day','item_cnt_avg_category_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_category_month')
# Average of overall items per category type sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','category_type'],'item_cnt_day','item_cnt_avg_categorytype_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_categorytype_month')
# Average of overall items per category subtype sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','category_sub-type'],'item_cnt_day','item_cnt_avg_categorysubtype_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_categorysubtype_month')
# Average of overall items per shop city sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_city'],'item_cnt_day','item_cnt_avg_shopcity_month')
monthly_data = calculate_lag(1,monthly_data,'item_cnt_avg_shopcity_month')

# #Months since last sale
monthly_data=monthly_data.sort_values(by=['shop_id','item_id','date_block_num'])
monthly_data['#months_since_last_sales'] = monthly_data['date_block_num']-monthly_data['date_block_num'].shift(1,axis=0)
monthly_data.loc[(monthly_data['shop_id']!=monthly_data['shop_id'].shift(1)) | (monthly_data['item_id']!=monthly_data['item_id'].shift(1)),'#months_since_last_sales']=0

# Dropping lag window values
learning_data = monthly_data[monthly_data['date_block_num'] < lag_window]

# Features to delete
features_to_drop = ['item_price','daily_revenue','item_name','item_category_name_translated','category_type','category_sub-type','shop_name_translated','shop_city','item_cnt_avg_month','item_cnt_avg_item_month','item_cnt_avg_shop_month','item_cnt_avg_category_month','item_cnt_avg_categorytype_month','item_cnt_avg_categorysubtype_month','item_cnt_avg_shopcity_month'];
learning_data = monthly_data.drop(features_to_drop, axis=1);
learning_data = learning_data.rename(columns={'item_cnt_day':'item_cnt_month'});
learning_data.fillna(0,inplace=True)