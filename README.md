# DSAI_FP(Kaggle Predict Future Sales)   
## google Slide 連結 

##  如何執行  
### 把資料集放進來  (六個檔案)
### 建環境(pipenv)  
進入 shell  
### 執行 data preprocess (寫成指令)  
### 執行 train(寫成指令)  
### 執行test 得到submission.csv(寫成指令)  

 
 

## 過程

###寫作業過程(增加feature 調模型參數 看importance刪減features)  
增加不同滯後shift
調模型參數(sampling)
藉由importance 刪掉一些對performance幫助不大的features
可以再加item price 滯後!!!

###資料前處理  
1.刪掉outlier
2.把shop_name看似同一間的資料,統一成同一間shop
3.新增更多features(shop_city, main_categories, sub_categories, mean(同月份不同商品))

###前33個月資料放進XGboost Training 
model參數
觀察各features importance 把importance較小的feature刪掉減少資料前處理的時間
圖*2+result+importance

###Test (submission.csv)  
對第35個月做prediction,並把結果輸出 submission.csv
圖*1
  


## 參考資料  

https://xgboost.readthedocs.io/en/latest/parameter.html  

https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  

https://www.kaggle.com/dlarionov/feature-engineering-xgboost  

https://www.kaggle.com/zhangyunsheng/xgboost/comments   

