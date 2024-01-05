from time import time
from os import os
""" This module provides functions for converting timestamps to and from human-readable time formats. """

'''
timeString = "2020-08-10_15h30" # 時間格式為字串
struct_time = time.strptime(timeString, "%Y-%m-%d_%Hh%M") # 轉成時間元組
time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
print(time_stamp)
'''

def JPGtoPNG():
    for file in os.listdir("./"+file_nom):   
        struct_time = time.strptime(file[:-4], "%Y-%m-%d_%Hh%M") # 轉成時間元組
        time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
        os.rename("./"+file_nom+"/"+file,"./"+file_nom+"/"+str(time_stamp)+".png")
        print(file[:-4]+"->"+str(time_stamp))
    return
    
""" Converts a time string to a timestamp. """
def TimeToTimestamps(Time):
    struct_time = time.strptime(Time, "%Y-%m-%d_%Hh%M") # 轉成時間元組
    time_stamp = int(time.mktime(struct_time)) # 轉成時間戳
    #print(time_stamp)
    return time_stamp
    
""" Converts a timestamp to a time string. """
def TimestampsToTime(Timestamp):
    t = time.localtime(int(Timestamp))
    timeStr = time.strftime("%Y-%m-%d_%Hh%M", t)
    return timeStr

if __name__ == "__main__":
    #file_nom=input("資料夾名稱：")
    #JPGtoPNG()
    #1596990600
    '''
    timeString = "2020-08-10_00h30" # 時間格式為字串
    print(TimeToTimestamps(timeString))
    '''
    
    Timestamp=1596990600
    print(TimestampsToTime(Timestamp))
    '''
    timestamp=1596990600



    #转换成localtime
    time_local = time.localtime(timestamp)
    #转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    print(dt)
    '''
