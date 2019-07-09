import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#此处的编码格式不能为UTF-8和gb2312。
train = pd.read_csv('D:/PythonWorks/vscode/XGBoost/Train_nyOWmfK.csv',encoding='ISO-8859-1')
test = pd.read_csv('D:/PythonWorks/vscode/XGBoost/Test_bCtAN1w.csv', encoding='ISO-8859-1')
#print("train.shape",train.shape)
#print("test.shape",test.shape)
#查看数据属性
#print(train.info()) 
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test],ignore_index=True)
#print("data.shape",data.shape)
#print(data.info())
#查看每个属性，为null的个数
#print(data.apply(lambda x: sum(x.isnull())))
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','source']
#查看var中的每个属性，分别有几类
for v in var:
    print('Frequency count for variable %s'%v)
    print(data[v].value_counts())
#查看属性city中有各种类别
print(len(data['City'].unique()))
#由于city中类别太多，所以直接drop掉，axis=1按照列删除，表示删除一整列
data.drop('City',axis=1,inplace=True)
#查看出生年月的前5行
#print(data['DOB'].head())
#对出生年月这一列进行转换，转换成具体的年龄,提取倒数后两个数字
data['Age'] = data['DOB'].apply(lambda x:115-int(x[-2:]))
#输出年龄的前5行
#print(data['Age'].head())
#由于有年龄这个属性，所以可以把DOB去掉
data.drop('DOB',axis=1,inplace=True)
#画出相应属性的点图
#data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')
#值为空的设置为1，值不为空设置为0,为了更清晰的看到缺失值
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x:1 if pd.isnull(x) else 0)
#输出前10行
print(data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10))
#drop掉
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)
#查看名字的总类别数
print(len(data['Employer_Name'].value_counts()))
#由于名字种类太多，所以直接去掉
data.drop('Employer_Name',axis=1,inplace=True)
#画出相应属性的点图
#data.boxplot(column='Existing_EMI',return_type='axes')
#输出该属性的均值，方差，最小值等信息
print(data['Existing_EMI'].describe())
#如果该属性值为空，则填充0
data['Existing_EMI'].fillna(0,inplace=True)
#如果为空置为1，不为空置为0
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x:1 if pd.isnull(x) else 0)
print(data[['Interest_Rate','Interest_Rate_Missing']].head(10))
data.drop('Interest_Rate',axis=1,inplace=True)
#因为没有直接关系，所以drop掉
data.drop('Lead_Creation_Date',axis=1,inplace=True)
#对于为空的值，用平均值填充
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x:1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x:1 if pd.isnull(x) else 0)
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)
data.drop('LoggedIn',axis=1,inplace=True)
#由于银行种类比较多，所以直接去掉
data.drop('Salary_Account',axis=1,inplace=True)
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x:1 if pd.isnull(x) else 0)
data.drop('Processing_Fee',axis=1,inplace=True)
#由于主要是前两类S122和S133,所以其他的归为others
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
#输出转换后的Source内容
#print(data['Source'].value_counts())
#输出每个属性为空的个数,目前就Disbursed属性有空值，此属性为标签
#print(data.apply(lambda x:sum(x.isnull())))
#查看属性的数据类型
#print(data.dtypes)
#由于属性中存在object类型，所以要把其进行转换
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#列出几个数据类型为object类型的属性
var_to_encode=['Device_Type','Filled_Form','Gender','Mobile_Verified','Source','Var1','Var2']
#把属性转换成数值类型
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
#把以上类型再次进行One-hot编码
data = pd.get_dummies(data,columns=var_to_encode)
#输出当前所有属性
#print(data.columns)
#把测试集合训练集分开
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop('source',axis=1,inplace=True)
#去掉Disbursed是因为，次属性是标签属性，inplace=True表示对原数组作出修改并返回一个新数组，原数组名对应的内存值直接改变
test.drop(['source','Disbursed'],axis=1,inplace=True)
#index表示是否写入行名称(索引)
train.to_csv('./train_modified.csv',index=False)
test.to_csv('./test_modified.csv',index=False)
print("train:",train.shape)
print("test:",test.shape)
#经过以上数据预处理后，除了ID之外，全部转变成可处理类型。
#plt.ylabel("ylabel")
#plt.xlabel("different datasets")
#plt.show()

