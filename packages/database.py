import mysql.connector
import gc
from mysql.connector import Error
def prepare(database_route,table):
    #新增資料庫
    connection = mysql.connector.connect(
        host='localhost',          # 主機名稱
        #database=database, # 資料庫名稱
        user='root',        # 帳號
        password='')  # 密碼
        
    #新增資料庫
    sql="CREATE DATABASE `"+database_route+"`"
    cursor = connection.cursor()
    cursor.execute(sql)
        
    #新增資料庫
    # 連接 MySQL/MariaDB 資料庫
    connection = mysql.connector.connect(
        host='localhost',          # 主機名稱
        database=database_route, # 資料庫名稱
        user='root',        # 帳號
        password='')  # 密碼
            
    #新增資料表
    sql="CREATE TABLE `"+table+"` (`Time` int(20) NOT NULL,`Rain` char(20) DEFAULT NULL,`Inundation` char(20) DEFAULT NULL,`Rain_Rate` float(20) DEFAULT NULL,`Inundation_Rate` float(20) DEFAULT NULL,`Inundation_Area` float(20) DEFAULT NULL,`Inundation_Depth` float(20) DEFAULT NULL,PRIMARY KEY (`Time`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("資料庫連線已關閉")
    # 確認資料有存入資料庫
    del connection
    gc.collect()

def insert_data(database_route, table, time,rain, inundation, rain_rate, inundation_rate,inundation_area,inundation_depth):
    # 連接 MySQL/MariaDB 資料庫
    connection = mysql.connector.connect(
        host='localhost',          # 主機名稱
        database=database_route, # 資料庫名稱
        user='root',        # 帳號
        password='')  # 密碼
    #connection.commit()
    # 新增資料
    sql = "INSERT INTO "+database_route+"."+table+" (Time, Rain, Inundation, Rain_Rate, Inundation_Rate,Inundation_Area,Inundation_Depth) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    new_data = (time, rain, inundation, rain_rate, inundation_rate,inundation_area,inundation_depth)
    cursor = connection.cursor()
    cursor.execute(sql, new_data)

    # 確認資料有存入資料庫
    connection.commit()
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("資料庫連線已關閉")
    del connection
    gc.collect()
        
def link_MYSQL(database_name):
    try:
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
        host='localhost',          # 主機名稱
        database=database_name, # 資料庫名稱
        user='root',        # 帳號
        password='')  # 密碼
        connection.commit()
        if connection.is_connected():
            # 顯示資料庫版本
            db_Info = connection.get_server_info()
            print("資料庫版本：", db_Info)
            # 顯示目前使用的資料庫
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print("目前使用的資料庫：", record)
            
    except Error as e:
        print("資料庫連接失敗：", e)

def add_database():
    try:
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            #database='test', # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼

        sql="CREATE DATABASE 'deux'"
        cursor = connection.cursor()
        cursor.execute(sql)
        #database='my_db'
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            database='deux', # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼
    except Error as e:
        print(e)
        
def add_datasheet(database,datasheet):

    try:
        '''
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            #database='test', # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼

        #新增資料庫
        sql="CREATE DATABASE `"+database+"`"
        cursor = connection.cursor()
        cursor.execute(sql)
        '''
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            database=database, # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼
        #新增資料表
        sql="CREATE TABLE `"+datasheet+"` (`Time` int(20) NOT NULL,`Rain` char(20) DEFAULT NULL,`Inundation` char(20) DEFAULT NULL,`Rain_Rate` float(20) DEFAULT NULL,`Inundation_Rate` float(20) DEFAULT NULL,`Inundation_Area` float(20) DEFAULT NULL,`Inundation_Depth` float(20) DEFAULT NULL,PRIMARY KEY (`Time`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
        cursor = connection.cursor()
        cursor.execute(sql)
        del connection
    except Error as e:
        print(e)
        
    gc.collect()
        
def add_data(database,datasheet,time,rain,inondation,rain_rate,inondation_rate,inundation_area,inundation_depth):
    try:
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            database=database, # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼
        connection.commit()
        try:
            # 新增資料
            sql = "INSERT INTO "+database+"."+datasheet+" (Time,Rain,Inundation, Rain_Rate, Inundation_Rate,Inundation_Area,Inundation_Depth) VALUES (%s, %s, %s, %s, %s,%s,%s);"
            new_data = (time, rain, inondation,rain_rate,inondation_rate,inundation_area,inundation_depth)
            cursor = connection.cursor()
            cursor.execute(sql, new_data)
        except Error as e1:
            #新增資料表
            sql="CREATE TABLE "+datasheet+" (`Time` char(10) NOT NULL,`Rain` char(10) DEFAULT NULL,`Inundation` char(10) DEFAULT NULL,`Rain_Rate` int(10) NOT NULL,`Inundation_Rate` int(10) NOT NULL,PRIMARY KEY (`Time`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
            cursor = connection.cursor()
            cursor.execute(sql)

            # 新增資料
            sql = "INSERT INTO "+database+"."+datasheet+" (Time,Rain,Inundation, Rain_Rate, Inundation_Rate) VALUES (%s, %s, %s, %s, %s);"
            new_data = (time, rain, inondation,rain_rate,inondation_rate)
            cursor = connection.cursor()
            cursor.execute(sql, new_data)
    except Error as eo:
        #新增資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            #database=database, # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼
        
        #新增資料庫
        sql="CREATE DATABASE `"+database+"`"
        cursor = connection.cursor()
        cursor.execute(sql)
        
        #新增資料庫
        # 連接 MySQL/MariaDB 資料庫
        connection = mysql.connector.connect(
            host='localhost',          # 主機名稱
            database=database, # 資料庫名稱
            user='root',        # 帳號
            password='')  # 密碼
            
        #新增資料表
        sql="CREATE TABLE "+datasheet+" (`Time` char(10) NOT NULL,`Rain` char(10) DEFAULT NULL,`Inundation` char(10) DEFAULT NULL,`Rain_Rate` int(10) NOT NULL,`Inundation_Rate` int(10) NOT NULL,PRIMARY KEY (`Time`)) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
        cursor = connection.cursor()
        cursor.execute(sql)

        # 新增資料
        sql = "INSERT INTO "+database+"."+datasheet+" (Time,Rain,Inundation, Rain_Rate, Inundation_Rate) VALUES (%s, %s, %s, %s, %s);"
        new_data = (time, rain, inondation,rain_rate,inondation_rate)
        cursor = connection.cursor()
        cursor.execute(sql, new_data)
        
        # 確認資料有存入資料庫
        connection.commit()
    
def update_data():
    try:
        # 更新資料
        sql = "UPDATE `persons` SET `city` = %s WHERE `persons`.`name` = %s;"
        cursor = connection.cursor()
        cursor.execute(sql, ("Rennes", "Jack"))
        # 確認資料有存入資料庫
        connection.commit()
    except Error as e:
        print(e)
        
def disconnect_database():
    try:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("資料庫連線已關閉")
    except Error as e:
        print(e)
        
if __name__ == "__main__":
    #link_MYSQL()
    #add_database()070753125
    #prepare('UPEM','A2')
    #add_datasheet(ville,commener)
    #prepare("1","test")
    #insert_data("database1", "personnel", 140000,"Yes", "Yes", 98.10, 100.00)
    #@profile
    insert_data("UPEM", "A2", 36000,"Yes", "Non", float(92.03), 0,0,0)
    insert_data("UPEM", "A2", 37000,"Yes", "Non", float(92.03), 0,0,0)
    insert_data("UPEM", "A2", 38000,"Yes", "Non", float(92.03), 0,0,0)
