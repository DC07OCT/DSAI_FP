import pandas as pd
import time
import os
import numpy as np

path=os.getcwd()
data_train=pd.read_csv(path+'/input_dataframe.csv')


#Train 值
train_input = data_train[data_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
#Train label
train_label = data_train[data_train.date_block_num < 33]['item_cnt_month']

#Validation 值
val_input = data_train[data_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)
#Validation label
val_label = data_train[data_train.date_block_num == 33]['item_cnt_month']
#Test 值
test_input = data_train[data_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)

from xgboost import XGBRegressor
#設定模型參數
param=dict( n_estimators=380,
                     max_depth=10,
                     colsample_bytree=0.5,
                     subsample=0.5,
                     learning_rate=0.01
           )
model = XGBRegressor(**param)

#訓練模型
model.fit(train_input.values, train_label.values,
          eval_metric="rmse",
          eval_set=[(train_input.values, train_label.values), (val_input.values, val_label.values)],
          verbose=True,
          early_stopping_rounds=50)#early_stopping_rounds is recommended to use 10% of your total iterations


importances = pd.DataFrame({'feature':data_train.drop('item_cnt_month', axis = 1).columns,'importance':np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances = importances[importances['importance'] > 0.01]
print('imp',importances)
#存模型
model.save_model('XGBr.model')
