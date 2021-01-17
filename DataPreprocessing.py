monthly_data = train.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day':'sum','item_price':'mean'}).reset_index(); # groupby automatically removes the column not mentioned, reset_index doesn't make the groupby columns as index
monthly_data = monthly_data.rename(columns={'date_block_num':'month_block_num','item_price':'mean_item_price','item_cnt_day':'item_cnt_month'});
# Combining all test points
monthly_data = pd.merge(test,monthly_data,on=['shop_id','item_id'],how='left') 
# Filling NaN
monthly_data=monthly_data.fillna(0)
# Sorting
monthly_data.sort_values(by=['shop_id','item_id','month_block_num'])
# Clipping
monthly_data['item_cnt_month'] = monthly_data['item_cnt_month'].clip(0,20)
# Combining all info in monthly_data table
monthly_data = pd.merge(monthly_data,items,on=['item_id'],how='inner')
monthly_data = pd.merge(monthly_data,categories,on=['item_category_id'],how='inner')
monthly_data = pd.merge(shops,monthly_data,on=['shop_id'],how='inner')


# Doing the same stuff for test
test_modified = pd.merge(test,items, on = ['item_id'],how='inner')
test_modified['month_block_num'] = 34
test_modified = test_modified.drop(['item_name'], axis=1)
test_modified = test_modified.set_index('ID')