# DSAI_FP(Kaggle Predict Future Sales)   
### [google Slide 連結(Presentatio用)]() 
### [google doc(詳細過程)]()

##  如何執行  
### 把資料集放進來  (六個檔案) 
dataset + input_data.csv+model+submission.csv
### 建環境(pipenv)  
在 cmd中執行  

    cd [目標資料夾]
輸入  

    pipenv shell
### 執行 data preprocess  
輸入    

    python DSAI_FP_PredFuture_preprocess.py
### 執行 train 
    python DSAI_FP_PredFuture_train.py
### 執行test 得到submission.csv  
    python DSAI_FP_PredFuture_test.py

 
 

## 過程

### Trail and error  
* 增加不同滯後shift  
* 調模型參數(sampling 和 early_stop_rounds)  
* 藉由importance 刪掉一些對performance幫助不大的features  
可以再加item price 滯後!!!(很難用)(item_price 對模型訓練有幫助 但第34個月沒有這筆資料所以test時 ˋ準確度會大跑掉)

### 資料前處理   
1.刪掉outlier
2.把shop_name看似同一間的資料,統一成同一間shop
3.新增更多features(shop_city, main_categories, sub_categories, mean(同月份不同商品)...)
4.[滯後特徵]將item_cnt_month 做shift指定月數後，依據‘item_id‘ 或’shop_id‘或’item_category_id’…將item_cnt_month取mean後併回原表格，做特徵。
5.將生成的滯後特徵中nan值，填成0

### 前33個月資料放進XGboost Training  
model參數  
觀察各features importance 把importance較小的feature刪掉減少資料前處理的時間  
圖*2+result(training+kaggle結果)+importance

### Test (得到submission.csv)    
對第35個月做prediction,並把結果輸出 submission.csv  
圖*1
  
### 詳細內容請看[google Doc]()

## 參考資料  

https://xgboost.readthedocs.io/en/latest/parameter.html  

https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  

https://www.kaggle.com/dlarionov/feature-engineering-xgboost  

https://www.kaggle.com/zhangyunsheng/xgboost/comments   

