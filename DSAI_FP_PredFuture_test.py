import pandas as pd
import os
path=os.getcwd()
submission=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sample_submission.csv')
data_train=pd.read_csv(path+'/input_dataframe.csv')
train_df=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sales_train.csv')

#Test 值
test_input = data_train[data_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# #更新特徵值'item_price_mean'刪掉
# group = train_df.groupby(['shop_id','item_id'])['item_price'].mean().rename('item_price_mean').reset_index()
# test_input = pd.merge(test_input.drop(['item_price_mean'], axis=1), group, on=['shop_id','item_id'], how='left').fillna(0)
# print(test_input)

from xgboost import XGBRegressor
#設定模型參數
# param=dict(n_estimators=350,
#                      max_depth=10,
#                      colsample_bytree=0.5,
#                      subsample=0.5,
#                      learning_rate=0.01
#            )
# model = XGBRegressor(**param)
model = XGBRegressor()
model.load_model('XGBr.model')#load model

#test
y_pred = model.predict(test_input.values)
print(y_pred)

#存csv
submission['item_cnt_month'] = y_pred
submission.to_csv('submission_final.csv', index=False)