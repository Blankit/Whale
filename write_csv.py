#coding:utf-8
import csv
import pandas as pd
file = 'class.txt'
fo = open(file)
# fw = open('class_ID.txt','w')
dic = {}
count  =10
for line in fo:
    line = line.strip()
    if line not in dic:
        dic[line] = 1
    else:
        dic[line] += 1
    count += 1

with open('num_every_class.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for k,v in dic.items():
        writer.writerow([k,v])
csvfile.close()
with open('calss_to_label.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    count = 0
    for k,_ in dic.items():
        writer.writerow([k,count])
        count += 1
csvfile.close()
# print(pd.read_csv('test.csv'))

fo.close()