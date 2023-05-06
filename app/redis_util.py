#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 20:28
# @Author  : Paulson
# @File    : testredis.py
# @Software: PyCharm
# @define  : function

import redis
from datetime import date


# 增加使用redis统计访问次数的功能



REDIS_HOST = "112.126.101.188"  # redis host
REDIS_PASSWORD = "ybsdemima@ybs"
MINIST_KEY = "MINIST"  # 总访问次数
TODAY_KEY = "TODAY"  # 今日日期
TODAY_TIME_KEY = "TODAY_TIME"  # 今日访问次数

# 初始化redis
try:
    r = redis.StrictRedis(host=REDIS_HOST, password=REDIS_PASSWORD)
except:
    print("redis 连接异常")
# 当前访问次数 查看日志中请求的次数，设置初始值  wc -l nohup.out
# 8390
# r.set(MINIST_KEY, 8390)
# r.set(TODAY_TIME_KEY, 1)
# 从redis中获取当前的访问次数


def inc_visit_num():
    count = int(r.get(MINIST_KEY).decode())
    if count < 100:
        print("redis中的访问量有问题，读取日志重新设置值")
        try:
            count = cal_line_num()
            r.set(MINIST_KEY, count)
        except:
            print("不是生产环境，请忽略")
        
    r.incr(MINIST_KEY)
    if (get_today().encode() == r.get(TODAY_KEY)):
        r.incr(TODAY_TIME_KEY)
    else:
        r.set(TODAY_KEY, get_today().encode())
        r.set(TODAY_TIME_KEY, 1)


def get_visit_num_all():
    return r.get(MINIST_KEY).decode()

def get_visit_num_today():
    return r.get(TODAY_TIME_KEY).decode()

def get_today():
    return date.today().strftime('%Y-%m-%d')

def cal_line_num():
    i = 0
    with open("./app/nohup.out", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            i = i+1
    print("LOG  总行数量: ", i)
    return i

if __name__ == '__main__':
    inc_visit_num() # 访问一次
    print(get_visit_num_all())
    print(get_visit_num_today())
   
