import pandas as pd
import os
path=os.getcwd()
submission=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sample_submission.csv')
data_train=pd.read_csv(path+'/input_dataframe.csv')
train_df=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sales_train.csv')

#Test 值
test_input = data_train[data_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)



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
