# Repeating FeatureGeneration notebook as a python script

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#### UPDATE in training 

data = pd.DataFrame({'shop_id' : [],'item_id' : [],'date_block_num' : []})
for i in range(34):
    sales_shops = train[train['date_block_num'] == i][['shop_id']].drop_duplicates(subset=['shop_id'])
    sales_items = train[train['date_block_num'] == i][['item_id']].drop_duplicates(subset=['item_id'])
    sales_shops['key']=0;
    sales_items['key']=0;
    train_block = sales_shops.merge(sales_items);  # creates a cartesian product of all (shop, item) pairs in test will all date_block_num values
    train_block = train_block.drop(['key'],axis=1);
    train_block['date_block_num']=i
    data = pd.concat([data, train_block], ignore_index=True, sort=False)

    
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


data['month'] = data['date_block_num']%12+1
data['year'] = 2013+data['date_block_num']//12
month_days = pd.Series([0,31,28,31,30,31,30,31,31,30,31,30,31]);
data['days_in_month'] = data['month'].map(month_days)

monthly_data = pd.merge(data,monthly_data,on=['shop_id','item_id','date_block_num','month','year','days_in_month'],how='left') 
monthly_data['ID']=-1

# Making test like train and adding -1 to the columns that will be deleted later
# test = test.drop(['ID'],axis=1)
test['date_block_num'] = 34;
test['item_cnt_day'] = -1;
test['item_price'] = -1;
test['daily_revenue'] = -1;
test['month'] = 11
test['year'] = 2015
test['days_in_month'] = 31
test.head()

monthly_data = pd.concat([monthly_data, test], ignore_index=True, sort=False)


monthly_data = pd.merge(monthly_data,items,on=['item_id'],how='inner')
monthly_data = pd.merge(monthly_data,categories,on=['item_category_id'],how='inner')
monthly_data = pd.merge(monthly_data,shops,on=['shop_id'],how='inner')


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

lag_window = 3;
# lag_window affects this
for i in ['item_cnt_day','item_price','daily_revenue']:
    for j in range(0,lag_window):
        monthly_data = calculate_lag(j+1,monthly_data,i)
        
        
train = pd.merge(train,items,on=['item_id'],how='inner')
train = pd.merge(train,categories,on=['item_category_id'],how='inner')
train = pd.merge(train,shops,on=['shop_id'],how='inner')


# Mean encoded features

# Average of overall items sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num'],'item_cnt_day','item_cnt_avg_month')
# Average of overall items per item sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','item_id'],'item_cnt_day','item_cnt_avg_item_month')
# Average of overall items per shop sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_id'],'item_cnt_day','item_cnt_avg_shop_month')
# Average of overall items per category sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','item_category_id'],'item_cnt_day','item_cnt_avg_category_month')
# Average of overall items per category type sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','category_type'],'item_cnt_day','item_cnt_avg_categorytype_month')
# Average of overall items per category subtype sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','category_sub-type'],'item_cnt_day','item_cnt_avg_categorysubtype_month')
# Average of overall items per shop city sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_city'],'item_cnt_day','item_cnt_avg_shopcity_month')

# Average of overall items per shop per item sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_id','item_id'],'item_cnt_day','item_cnt_avg_shopitem_month')
# Average of overall items per shop city per item sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_city','item_id'],'item_cnt_day','item_cnt_avg_shopcityitem_month')
# Average of overall items per shop cuty per item category type sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_city','category_type'],'item_cnt_day','item_cnt_avg_shopcitytype_month')
# Average of overall items per shop cuty per item category sub-type sold in a month
monthly_data = create_mean_features(train,monthly_data,['date_block_num','shop_city','category_sub-type'],'item_cnt_day','item_cnt_avg_shopcitysubtype_month')
                                           

for j in range(1,4):

    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_item_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shop_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_category_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_categorytype_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_categorysubtype_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shopcity_month')

    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shopitem_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shopcityitem_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shopcitytype_month')
    monthly_data = calculate_lag(j,monthly_data,'item_cnt_avg_shopcitysubtype_month')

    
lag_window2 = 3;
# Creating trend features

# delta price
monthly_data = create_mean_features(train,monthly_data,['item_id'],'item_price','item_avg_price')
monthly_data = create_mean_features(train,monthly_data,['item_id','date_block_num'],'item_price','item_monthly_avg_price')
for j in range(0,lag_window2):
    monthly_data = calculate_lag(j+1,monthly_data,'item_monthly_avg_price')
    monthly_data['delta_price_lag_' + str(j+1) ] = (monthly_data['item_monthly_avg_price_lag' + str(j+1)]- monthly_data['item_avg_price'] ) / monthly_data['item_avg_price']

# delta revenue
monthly_data = create_mean_features(train,monthly_data,['shop_id'],'daily_revenue','shop_avg_revenue')
monthly_data = create_mean_features(train,monthly_data,['shop_id','date_block_num'],'daily_revenue','shop_monthly_avg_revenue')
for j in range(0,lag_window2):
    monthly_data = calculate_lag(j+1,monthly_data,'shop_monthly_avg_revenue')
    monthly_data['delta_revenue_lag_' + str(j+1) ] = (monthly_data['shop_monthly_avg_revenue_lag' + str(j+1)]- monthly_data['shop_avg_revenue'] ) / monthly_data['shop_avg_revenue']

monthly_data = monthly_data.replace([np.inf, -np.inf], np.nan)



monthly_data["item_shop_first_sale"] = monthly_data["date_block_num"] - monthly_data.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
monthly_data["item_first_sale"] = monthly_data["date_block_num"] - monthly_data.groupby(["item_id"])["date_block_num"].transform('min')


# #Months since last sale
monthly_data=monthly_data.sort_values(by=['shop_id','item_id','date_block_num'])
monthly_data['#months_since_last_sales'] = monthly_data['date_block_num']-monthly_data['date_block_num'].shift(1,axis=0)
monthly_data.loc[(monthly_data['shop_id']!=monthly_data['shop_id'].shift(1)) | (monthly_data['item_id']!=monthly_data['item_id'].shift(1)),'#months_since_last_sales']=0

# To preserve order of X_test
monthly_data = monthly_data.sort_values(by=['ID'])
monthly_data = monthly_data.drop(columns=['ID'])

# Dropping lag window values
# learning_data = monthly_data[monthly_data['date_block_num'] >= lag_window]

# Not dropping lagged values gives better result
learning_data = monthly_data

