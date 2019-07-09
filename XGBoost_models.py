import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection,metrics
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

#关于GridSearchCV网格搜索这个模块，新版本是在model_selection里面
rcParams['figure.figsize'] = 12,4
#载入之前经过预处理后的数据 
train = pd.read_csv('./train_modified.csv')
test = pd.read_csv('./test_modified.csv')
#标签属性
target='Disbursed'
IDcol='ID'
#列出该属性所有类别和其对应的数量
print(train['Disbursed'].value_counts())
#early_stopping_rounds最早结束循环是50
def modelfit(alg,dtrain,dtest,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):
    if useTrainCV:
        #表示获取xgb的参数
        xgb_param = alg.get_xgb_params()
        #把数据分成其他属性的数据和标签数据，类型转换为二进制形式
        xgtrain = xgb.DMatrix(dtrain[predictors].values,label=dtrain[target].values)
        #只保留测试数据中其他属性的数据，标签数据是要进行预测的
        xgtest = xgb.DMatrix(dtest[predictors].values)
        #xgb.cv用于交叉验证，其中nfold表示输出最好的轮数
        cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,metrics='auc',early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    #fit数据
    alg.fit(dtrain[predictors],dtrain['Disbursed'],eval_metric='auc')
    #预测
    dtrain_predictions = alg.predict(dtrain[predictors])
    #获取预测为真的概率值
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    print("Model Report")
    #输出精度，真实值和预测值之间的比值
    print("Accuracy:%.4g"% metrics.accuracy_score(dtrain['Disbursed'].values,dtrain_predictions))
    #AUC就是正整例、假正例等之间的比值关系
    print("AUC Score (Train):%f"% metrics.roc_auc_score(dtrain['Disbursed'],dtrain_predprob))
    #feat_imp2 = alg.get_booster().get_fscore()
    #print("feat_imp2:",feat_imp2)

    #feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar',title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    plot_tree(alg)
    plt.show()

if __name__=='__main__':
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    xgb1 = XGBClassifier(
    learning_rate = 0.01,
    n_estimators=5000,
    max_depth=5,
    min_child_weight=5,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
    modelfit(xgb1,train,test,predictors)
    """
    param_test1={
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
    }
    #通过grid serach 这种组合的形式来筛选最合适的属性
    gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,n_estimators=128,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=27),param_grid=param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
    gsearch1.fit(train[predictors],train[target])
    print("score:",gsearch1.grid_scores_)
    print("params:",gsearch1.best_params_)
    print("best_score:",gsearch1.best_score_)
    """
