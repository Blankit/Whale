#coding:utf-8
import pandas as pd

'''
1. 用pandas将数据存到csv中。
2. 将鲸鱼的类别转成数字，用于计算损失
3. index = df[df.cls=='w_8e4abc9'].index.tolist()
label = int(index[0])
返回类别'w_8e4abc9'鲸鱼的标签，并转换成int类型
这里，选用的标签是索引值
4. 类别与标签的对应列表在class_label.csv中
'''

file = 'class.txt'
fo = open(file)
# fw = open('class_ID.txt','w')
dic = {}
count  = 0
for line in fo:
    line = line.strip()
    if line not in dic:
        dic[line] = 1
    else:
        dic[line] += 1
    count += 1
df = pd.DataFrame(data=[dic.keys()])
df = df.T

df.index.name='label'
# df.index+=1
df.columns=['cls']
df.to_csv('class_label.csv', header=True)

