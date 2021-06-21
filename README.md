# DSAI_FP(Kaggle Predict Future Sales)   
### [google Slide 連結(Presentation用)](https://docs.google.com/presentation/d/1as8rI73T3gprqfWMLaZYT08saAlG_JB9HK06ZM1LIZ0/edit?usp=sharing) 
### [google doc(詳細過程)](https://drive.google.com/file/d/16GHQbTxprT90t_UXaKu67YR6xLcOGU1Y/view?usp=sharing)

##  如何執行  
### 下載Kaggel資料集  
https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data  
準備成   
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/prepare_1.png)
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

### 資料前處理   
1.刪掉outlier  
2.把shop_name看似同一間的資料,統一成同一間shop    
3.新增更多features(shop_city, main_categories, sub_categories, mean(同月份不同商品)...)  
4.[滯後特徵]將item_cnt_month 做shift指定月數後，依據‘item_id‘ 或’shop_id‘或’item_category_id’…將item_cnt_month取mean後併回原表格，做特徵。  
5.將生成的滯後特徵中nan值，填成0  

### 前33個月資料放進XGboost Training  
model參數   
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/model_1.png)  
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/model_2.png)  
Validation Result
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/result_1.png)  

觀察各features importance,把importance較小的feature刪掉,減少資料前處理的時間   
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/importance.png)  

### Test (得到submission.csv)    
對第35個月做prediction,並把結果輸出 submission.csv  
![image](https://github.com/DC07OCT/DSAI_Final-Project/blob/main/Figures/result_2.png)  

### Trail and error  
* 增加不同滯後shift  
* 調模型參數(sampling 和 early_stop_rounds)  
* 藉由importance 刪掉一些對performance幫助不大的features  


  
### 詳細內容請看[google Doc](https://drive.google.com/file/d/16GHQbTxprT90t_UXaKu67YR6xLcOGU1Y/view?usp=sharing)

## 參考資料  

https://xgboost.readthedocs.io/en/latest/parameter.html  

https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  

https://www.kaggle.com/dlarionov/feature-engineering-xgboost  

https://www.kaggle.com/zhangyunsheng/xgboost/comments   

