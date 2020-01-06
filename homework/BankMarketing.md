# Bank Marketing



## 1 实验目的

​	通过机器学习方法，对采集的数据进行学习，进而预测客户是否将认购定期存款。



## 2 数据描述



### 2.1 数据来源

​	数据url：•[http://](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)[archive.ics.uci.edu/ml/datasets/Bank+Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

​	数据采集自葡萄牙银行机构的直接营销活动。市场营销活动使用的是电话方式。

### 2.2 数据集信息

​	数据总共有四个数据集：

1. 带所有示例的bank-additional-full.csv（45211）和20个输入，按日期排序（从2008年5月到2010年11月），非常接近[Moro et al。，2014]中分析的数据。
2. bank-additional.csv，其中有10％的示例（4119），从1）中随机选择，还有20个输入。
3. bank-full.csv，包含所有示例和17个输入，按日期排序（此数据集的旧版本，输入较少）。
4. bank.csv，其中有10％的示例和17个输入，是从3个随机选择的（此数据集的旧版本，输入较少）。 

###  2.3 属性信息

 输入变量：

**＃银行客户数据：**

1. age：年龄（数字）

2. job：职务类型（类别：“管理员”，“蓝领”，“企业家”，“女仆”，“管理”，“退休” ，“自雇”，“服务”，“学生”，“技术员”，“待业”，“未知”）

3. martial：婚姻状况（类别：“离婚”，“已婚”，“单身”，“未知”） ';注意：“离婚”是指离婚或丧偶）

4. education：教育情况（类别：“ unknown", "secondary", "primary", "tertiary")

5. default：信用违约吗？（类别：“否”，”是”，“未知”） 

6. balance：年收入（欧元，数字）

7. housing：有住房贷款吗？（分类：“否”，“是”，“未知”）

8. loan：有个人贷款吗？（类别：“否”，“是”，“未知”）

9. contact：联系人通讯类型（类别：“蜂窝”，“电话”）

10. day：一月中的最后联系日（数字）

11. 月：一年中的最后联系人月份（类别：“ jan”，“ feb”，“ mar'，...，'nov'，'dec'）

12. duration：最后一次接触持续时间，以秒为单位（数字）。
    **＃其他属性：**

13. compaign：在该广告活动期间和该客户执行的联系次数（数字，包括最后一次联系）

14. pdays：从上一次广告活动最后一次联系客户以来经过的天数（数字； 999表示客户不是先接触）

15. previous：这次竞选前，此客户进行接触次数（数字）

16. poutcome：以前营销活动的结果（分类的结果：“失败”，“不存在”，“成功”）

    **#输出变量（期望值）目标）：**

17. y-客户是否已订阅定期存款？（二进制：“是”，“否”） 



## 3 数据预览

### 3.1 数据预览

```
data.head(10)
```

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578287537103.png" alt="1578287537103" style="zoom:80%;" />

只打印了前10个数据，可以看出，在最后一项是否订阅定期存款中，都是 `no`，这意味着该数据集的输出是不平衡的。

部分数据中还有 `unknown`，说明部分数据是未知的。



### 3.2 数据描述

```python
data.info()
```

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577971449533.png" alt="1577971449533" style="zoom:80%;" />

没有缺失数据，不需要填充。

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578287722955.png" alt="1578287722955" style="zoom: 67%;" />

总共有 45211 个数据，

### 3.3 数据分析

#### 3.3.1 数据分布情况

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578287907816.png" alt="1578287907816" style="zoom:80%;" />

可以看出，年龄主要集中在 25~60 岁之间。年收入大部分都在 10000 欧元以下，有的甚至为负值。最后一次的接触时间基本在 1000s 以下。从 previous 的分布可以看出，大部分客户都是第一次接触。



#### 3.3.2 职业相关分析

##### 3.3.2.1 职业分布

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577969152097.png" alt="1577969152097" style="zoom: 67%;" />

可以看出，样本中管理人员最多，学生和未就业人员最少。

##### 3.3.2.2 职业与年龄

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577969493826.png" alt="1577969493826" style="zoom: 67%;" />

##### 3.3.2.3 职业与收入

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577971380249.png" alt="1577971380249" style="zoom: 67%;" />



#### 3.3.3 受教育情况分析

##### 3.3.3.1 受教育情况分布

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577969899774.png" alt="1577969899774" style="zoom:80%;" />

可以看出接收大学教育的人数是最多的。

##### 3.3.3.2 受教育情况与年龄分布

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577970184392.png" alt="1577970184392" style="zoom:80%;" />

发现接受较好教育的人年龄偏小，而受教育水平相对较低的人年龄偏大。反映了随着社会的发展，葡萄牙的教育水平在提升。

##### 3.3.3.3 受教育与收入

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577971610033.png" alt="1577971610033" style="zoom:80%;" />

## 4 订阅定期存款的相关性分析

### 4.1 整体的相关性分析

![1577973156836](C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1577973156836.png)可以看出，与是否订阅定期存款的相关性最大的数值类型是 duration，而相关性最小的是年龄。

下面进一步分析通话时间对是否订阅存款业务的具体影响：

下面是订阅定期存款用户的描述：

```python
data_yes = data[data['y'] == 'yes']
data_no = data[data['y'] == 'no']
print(data_yes.describe())
print(data_no.describe())
```

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578288741153.png" alt="1578288741153" style="zoom:80%;" />

从上图中的数据对比可以看出，订阅存款人的平均年龄比没有订阅的大1岁，其平均年收入也比没有订阅的人多 500 欧元。

从上面两个图都可以看出，持续时间越长就越有可能订阅定期存款。订阅存款人的平均持续时间是为订阅定期存款人持续时间的两倍多。

pdays 越大，订阅存款的可能性也越大。订阅存款人的 pdays 接近为订阅人的两倍。

4.2 婚姻情况与订阅存款的关系

```python
data_yes.marital.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
data_no.marital.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
```

![1578289548567](C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578289548567.png)

左边是为订阅存款人的婚姻情况分布，右边是订阅存款人的婚姻情况分布。

可以看出：单身人士偏向于订阅存款，而结婚人士偏向于不订阅存款。离婚人士在而这种的偏向性不明显。



## 5 预处理方法

### 5.1 缺失值处理

对于缺失值较少的变量部分，直接将缺失值所在行数据删除。

```python 
if data[data[i] == 'unknown']['y'].count < 500:
    data = data[data[i] != 'unknown']
```

### 5.2 分类变量数值化

对于 `education`, `job`, `marital`, `default`, `housing`, `loan` 等变量，由于其不是数值，sklearn 库的机器学习算法无法进行处理，因此需要对其进行数值化。

#### 5.2.1 二分类的数值化

对于 `default`, `housing`, `loan`等变量直接进行二元编码：

```pyhton
def encode_bin_attrs(data, bin_attrs):
	for i in bin_attrs:
		data[i] = data[i].map({'yes':1, 'no':0, 'unknown':'unknown'})
```



#### 5.2.2 `education` 数值化

`education`不包括`unknown`的情况下总共有三类：`secondary`, `primary`, `tertiary`

由于是三种教育类型是有序的，因此采用连续赋值：

```pyhton
def encode_edu_attrs(data):
	data['education'] = data['education'].map({'secondary': 1, 'primary': 2, 'tertiary': 3})
	return data
```

#### 5.2.3 无序变量数值化

由于变量中的各个值之间没有顺序，因此不可按照`education`变量数值化方式，应当使用哑变量。这些变量是：job, marital, contact, month等。

```python
def encode_cate_attrs(data, cate_attrs):
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i + '_' + str(x))
        data = pd.concat([data, dummies_df], axis=1)
        data = data.drop(i, axis=1)
    return data
```

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578294308383.png" alt="1578294308383" style="zoom:80%;" />

哑变量的形式如上图所示：

对于婚姻状况，原来是`marital`变量中包含三种情况，现在变成了三个变量，如果是其中某一种情况，则现在对应情况下的变量值为1， 否则为0.

#### 5.2.4 数值特征归一化

```python
std = data[i].std()
if std != 0:
	mean = data[i].mean()
	data[i] = (data[i] - mean) / std
else:
	data = data.drop(i, axis=1)
```

对于每一个数值特征变量，其结果为减去平均值然后除以标准差。

#### 5.2.5 填充缺失值

若特征量的内部缺失值个数少于500，则保留并进行填充。填充方式是使用随机森林法。

对于任何一个被处理的特征，首先依据该特征将数据集分成该特征值不缺失的数据集和特征值缺失的数据集。对于不缺失的数据集作为训练集，缺失该特征的数据集作为测试集。其中该特征为输出的结果。首先基于该训练集训练随机森林模型。训练好后使用测试集的 `test_x` 进行测试，输出的预测值作为相对应的缺失值的替换值。

处理方式如下图所示：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578314065686.png" alt="1578314065686" style="zoom:80%;" />

#### 5.2.6 保存预处理结果

处理完成后，将结果保存到 `processed_data.csv` 文件中。使用的是 `pandas` 的 `to_csv` 方法。需要使用的时候直接将其读取出来即可。



## 6 模型训练和评估

### 6.1 数据集划分

将数据集划分成两部分，分别是训练集和测试集。训练集用来训练模型，测试集用来测试模型的泛化性能和准确性。

### 6.2 训练集重采样

由于数据集的输出结果并不均衡，从图中也可以看出，大部分都是 `no`，少部分是 `yes`。如果直接进行训练，那么模型倾向于输出 `no` 的结果。对于输出结果为 `yes` 的样例的准确性比较差。

因此，需要对数据集进行认为的划分，增加结果为 `yes` 的样例，减少结果为 `no` 的样例。这样模型才能更准确地识别出结果为 `yes` 的情况。而这个也是银行营销数据采集的意义，即提高营销的效率，更精准的识别需要订阅定期存款的用户。

主要测用两种重采样方式：

- 过采样： 采样处理不平衡数据的最常用方法，基本思想就是通过改变训练数据的分布来消除或减小数据的不平衡。过抽样方法通过增加少数类样本来提高少数类的分类性能 ，最简单的办法是简单复制少数类样本，缺点是可能导致过拟合，没有给少数类增加任何新的信息，泛化能力弱。改进的过抽样方法通过在少数类中加入随机高斯噪声或产生新的合成样本等方法。  
- 欠采样： 欠采样方法通过减少多数类样本来提高少数类的分类性能，最简单的方法是通过随机地去掉一些多数类样本来减小多数类的规模，缺点是会丢失多数类的一些重要信息，不能够充分利用已有的信息。 

本文采用 `Smote` 算法通过增加新的样本进行过采样，采用随机去掉一些多数类样本进行欠采样。

 Smote算法的基本思想是对于少数类中每一个样本x，以欧氏距离为标准计算它到少数类样本集中所有样本的距离，得到其k近邻。然后根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本x，从其k近邻中随机选择若干个样本，构建新的样本。针对本实验的数据，为防止新生成的数据噪声过大，新的样本只有数值型变量真正是新生成的，其他变量和原样本一致。 

### 6.3 模型训练和评估

本文选用的模型为：决策树、逻辑回归和随机森林

论文引用：

 [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

#### 6.3.1 随机森林

下面是随机森林的`precision-recall` 曲线：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322235741.png" alt="1578322235741" style="zoom:67%;" />

Accuracy: 0.876

下面是该模型的报告：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322283457.png" alt="1578322283457" style="zoom:80%;" />



#### 6.3.2 决策树

下面是决策树的`precision-recall` 曲线：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322356817.png" alt="1578322356817" style="zoom:67%;" />

Accuracy: 0.801

下面是该模型的报告：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322382345.png" alt="1578322382345" style="zoom:80%;" />



#### 6.3.3 逻辑回归

下面是逻辑回归的`precision-recall` 曲线：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578321726037.png" alt="1578321726037" style="zoom:67%;" />

 Accuracy: 0.816

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322444186.png" alt="1578322444186" style="zoom: 67%;" />![1578322502666](C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322502666.png)



#### 6.3.4 KNN

下面是KNN的`precision-recall` 曲线：

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322511326.png" alt="1578322511326" style="zoom:67%;" />

Accuracy: 0.855

<img src="C:\Users\WuJianeng\AppData\Roaming\Typora\typora-user-images\1578322533218.png" alt="1578322533218" style="zoom:80%;" />

