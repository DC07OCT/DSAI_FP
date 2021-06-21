
import pandas as pd
from sklearn import preprocessing
import numpy as np
from itertools import product
import datetime
import time
import os


def fill_nan(df):#把nan的值 填成0
    for col in df.columns:
        if ('_shift_' in col) & (df[col].isna().any()):
            df[col].fillna(0, inplace=True)
    return df


def downcast_dtypes(df):#做 data type轉換

    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols] = df[int_cols].astype(np.int16)

    return df
def generate_shift(train, months, shift_column):#train是DF,month是要shift的月數,shift_column要移動的數值col  #把shift_column的資料  時間向後shift month個月再併入原DF
    for month in months:
        # 建立shift(shift) features
        train_shift = train[['date_block_num', 'shop_id', 'item_id', shift_column]].copy()
        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', shift_column+'_shift_'+ str(month)]#改col name
        train_shift['date_block_num'] += month#做shift
        #把此特徵train_shift merge回原本dataframe
        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return train

#讀取資料
path=os.getcwd()
train_df=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sales_train.csv')
id_table=pd.read_csv(path+'/competitive-data-science-predict-future-sales/test.csv')
shop=pd.read_csv(path+'/competitive-data-science-predict-future-sales/shops.csv')
items=pd.read_csv(path+'/competitive-data-science-predict-future-sales/items.csv')
item_cate=pd.read_csv(path+'/competitive-data-science-predict-future-sales/item_categories.csv')
submission=pd.read_csv(path+'/competitive-data-science-predict-future-sales/sample_submission.csv')


#要先處理train_df的outlier(針對銷售價格 跟 item_cnt_day)
train_df=train_df[train_df['item_price']>0]
train_df=train_df[train_df['item_price']<30000]
train_df=train_df[train_df['item_cnt_day']>0]
train_df=train_df[train_df['item_cnt_day']<1000]
#print(train_df) #[2265266 rows x 6 columns]
#把幾個shop名稱相近看似為重複登記的資料合併{0,57}{1,58}{10,11}
train_df.loc[train_df['shop_id'] == 0, 'shop_id'] = 57
id_table.loc[id_table['shop_id'] == 0, 'shop_id'] = 57#並到57(train和test都要做)
train_df.loc[train_df['shop_id'] == 1, 'shop_id'] = 58
id_table.loc[id_table['shop_id'] == 1, 'shop_id'] = 58#並到58
train_df.loc[train_df['shop_id'] == 10, 'shop_id'] = 11
id_table.loc[id_table['shop_id'] == 10, 'shop_id'] = 11#並到11
#多新增shop_city資訊
shop_cities = shop['shop_name'].str.split(' ').str[0]#取出城市名稱
shop['city'] = shop_cities#新增一欄'city'
shop.loc[shop.city == '!Якутск', 'city'] = 'Якутск'#合併{'!Якутск','Якутск'}
label_encoder = preprocessing.LabelEncoder()#工具
shop['city_id'] = label_encoder.fit_transform(shop['city'])#轉換成city_id
#'shop_name'和'city'删除 剩 shop_id ,city_id
shop = shop.drop(['shop_name'], axis = 1)
#print(shop)

#前處理item_categories表(增加欄位,main_category,sub_category,並轉換main_category_id,sub_category_id)
cats_ = item_cate['item_category_name'].str.split('-')
#提取main_category(前半字符)
item_cate['main_category'] = cats_.map(lambda row: row[0].strip())  # 提取前面的字符，用strip()用於删除非字符部分
#提取sub_category（若無 就填入main category）
item_cate['sub_category'] = cats_.map(lambda row: row[1].strip() if len(row) > 1 else row[0].strip())
#對'main_category' 'sub_category'進行id編碼
item_cate['main_category_id'] = label_encoder.fit_transform(item_cate['main_category'])
item_cate['sub_category_id'] = label_encoder.fit_transform(item_cate['sub_category'])
#print(item_cate)


#把date轉換成datetime格式
train_df["date"]=[datetime.datetime.strptime(item,'%d.%m.%Y') for item in train_df["date"]]#sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
#將Train data 33格月份中的(Shop-Item-data) 做笛卡爾元組轉換 建立DF
months = train_df['date_block_num'].unique()
cartesian = []
for month in months:
    shops_in_month = train_df.loc[train_df['date_block_num'] == month, 'shop_id'].unique()
    items_in_month = train_df.loc[train_df['date_block_num'] == month, 'item_id'].unique()
    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))
cartesian_df = pd.DataFrame(np.vstack(cartesian), columns=['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)
#print('cartesian',cartesian_df)
#把train.csv中item_cnt_day轉換成item_cnt_month(sum)
x = train_df.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()
#將train data中有值的項目對應笛卡爾元組 填入item_cnt_month的值 合併這兩個DF 且x中沒有對應值的部分填0
data_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
#根據題目要求 限制到[0,20]之内 超出範圍 當作 0 20
data_train['item_cnt_month'] = np.clip(data_train['item_cnt_month'], 0, 20)
#使用sort_values對DF 按'date_block_num','shop_id','item_id元素做内部排序
data_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)
#print(data_train)
#刪除不用表
del shop_cities
del x
del cartesian_df
del cartesian
#把表格加上date_block_num=34的列 把表格目標位置準備好(暂定為0）
id_table.insert(loc=3, column='date_block_num', value=34)
id_table['item_cnt_month'] = 0  # 暫定為零
#把data_train嫁接加上'date_block_num'=34的列們
data_train = data_train.append(id_table.drop('ID', axis = 1)) #因為id_table中有ID欄但data_train中沒有 所以要把他drop掉
#merge到目標表 根據'shop_id'
data_train = pd.merge(data_train, shop, on=['shop_id'], how='left')
#merge到目標表 根據'item_id'
data_train = pd.merge(data_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')
#merge到目標表 根據'item_category_id'
data_train = pd.merge(data_train,  item_cate.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')
#删除非數值的列
data_train.drop(['main_category','sub_category','city'],axis=1,inplace=True)
#所有資料都整合到data_train
del shop
del items
del item_cate
del id_table
del train_df
#轉換data_type(int16)
data_train = downcast_dtypes(data_train)
#增加月休息天數feature
data_train['month'] = data_train['date_block_num'] % 12
holiday_dict = {0: 6,
                1: 3,
                2: 2,
                3: 8,
                4: 3,
                5: 3,
                6: 2,
                7: 8,
                8: 4,
                9: 8,
                10: 5,
                11: 4}
data_train['holidays_in_month'] = data_train['month'].map(holiday_dict)

# #增加特徵(item_price)
# group = data_train.groupby(['shop_id','item_id'])['item_price'].mean().rename('item_price_mean').reset_index()#按月份 分'item_id' 算'item_cnt_month'的mean
# data_train = pd.merge(data_train, group, on=['shop_id','item_id'], how='left')#對應'date_block_num', 'item_id' 新增欄'item_month_mean'資訊
# data_train['item_price_meandiff']=data_train['item_price']-data_train['item_price_mean']

#生成滯後特徵(單純把item_cnt_month 做shift)
data_train = generate_shift(data_train, [1,2,3,4,5], 'item_cnt_month')
#data_train = generate_shift(data_train, [1,2,3,4,5,6,12], 'item_cnt_month')#[1,2,3,4,5,6,12]
#生成滯後特徵(以'item_id'先做mean(item_cnt_month) 再做shift)
group = data_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()#按月份 分'item_id' 算'item_cnt_month'的mean
data_train = pd.merge(data_train, group, on=['date_block_num', 'item_id'], how='left')#對應'date_block_num', 'item_id' 新增欄'item_month_mean'資訊
data_train = generate_shift(data_train, [1,2,6], 'item_month_mean')#把mean(item_cnt_month') 做1,2,6個月shift作為新的特徵
#data_train = generate_shift(data_train, [1,2,3,6,12], 'item_month_mean')#[1,2,3,6,12]
data_train.drop(['item_month_mean'], axis=1, inplace=True)#删除不需要的'item_month_mean'属性
#生成滯後特徵(以'shop_id'先做mean(item_cnt_month) 再做shift)
group = data_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()#按月份 分'shop_id' 算'item_cnt_month'的mean
data_train = pd.merge(data_train, group, on=['date_block_num', 'shop_id'], how='left')
data_train = generate_shift(data_train, [1,2], 'shop_month_mean')#[1,2,3,6,12]
#data_train = generate_shift(data_train, [1,2,3,6,12], 'shop_month_mean')#[1,2,3,6,12]
data_train.drop(['shop_month_mean'], axis=1, inplace=True)
#生成滯後特徵(以'shop_id', 'item_category_id'先做mean(item_cnt_month) 再做shift)
group = data_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()#按月份 分'shop_id'+'item_category_id' 算'item_cnt_month'的mean
data_train = pd.merge(data_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
data_train = generate_shift(data_train, [1, 2], 'shop_category_month_mean')#[1, 2]
data_train.drop(['shop_category_month_mean'], axis=1, inplace=True)#刪欄
#生成滯後特徵(以'main_category_id'先做mean(item_cnt_month) 再做shift)
group = data_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()#按月份 分'main_category_id' 算'item_cnt_month'的mean
data_train = pd.merge(data_train, group, on=['date_block_num', 'main_category_id'], how='left')
data_train = generate_shift(data_train, [12], 'main_category_month_mean')#[1]
#data_train = generate_shift(data_train, [1,12], 'main_category_month_mean')#[1,12]
data_train.drop(['main_category_month_mean'], axis=1, inplace=True)
#生成滯後特徵(先'sub_category_id'做mean(item_cnt_month) 再做shift)
group = data_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()#按月份 分'sub_category_id' 算'item_cnt_month'的mean
data_train = pd.merge(data_train, group, on=['date_block_num', 'sub_category_id'], how='left')
data_train = generate_shift(data_train, [1], 'sub_category_month_mean')#[1]
data_train.drop(['sub_category_month_mean'], axis=1, inplace=True)

#print(data_train)#29columns
#print(sum(data_train.count()))#234031423

#沒有數據的樣本用0填充
data_train = fill_nan(data_train)
#存DF
data_train.to_csv('input_dataframe.csv', index=False)

#
# print('start',time.ctime())
#
# #Train 值
# train_input = data_train[data_train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
# #Train label
# train_label = data_train[data_train.date_block_num < 33]['item_cnt_month']
#
# #Validation 值
# val_input = data_train[data_train.date_block_num == 33].drop(['item_cnt_month'], axis=1)
# #Validation label
# val_label = data_train[data_train.date_block_num == 33]['item_cnt_month']
#
# #Test值
# test_input = data_train[data_train.date_block_num == 34].drop(['item_cnt_month'], axis=1)
#
# from xgboost import XGBRegressor
# #設定模型參數
# param=dict(n_estimators=3000,
#                      max_depth=10,
#                      colsample_bytree=0.7,
#                      subsample=0.7,
#                      learning_rate=0.01
#
#            )
# model = XGBRegressor(**param)
#
# #訓練模型
# model.fit(train_input.values, train_label.values,
#           eval_metric="rmse",
#           eval_set=[(train_input.values, train_label.values), (val_input.values, val_label.values)],
#           verbose=True,
#           early_stopping_rounds=50)#early_stopping_rounds is recommended to use 10% of your total iterations
# #存模型
# model.save_model('XGBr.model')
#
# #Predict結果
# y_pred = model.predict(test_input.values)
# print(y_pred)
# print('End',time.ctime())
# # #特徵重要性
# # importances = pd.DataFrame({'feature':data_train.drop('item_cnt_month', axis = 1).columns,'importance':np.round(model.feature_importances_,3)})
# # importances = importances.sort_values('importance',ascending=False).set_index('feature')
# # importances = importances[importances['importance'] > 0.01]
# # print('imp',importances)
# # importances.plot(kind='bar',
# #                  title = 'Feature Importance',
# #                  figsize = (8,6),
# #                  grid= 'both')
# # plot_feature_importance(np.round(model.feature_importances_,3),data_train.drop('item_cnt_month', axis = 1).columns,'XG BOOST')
#
# #存csv
# submission['item_cnt_month'] = y_pred
# #print(y_pred)
# submission.to_csv('submission.csv', index=False)