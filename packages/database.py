import mysql.connector

# 連接 MySQL/MariaDB 資料庫
connection = mysql.connector.connect(
    host='localhost',          
    user='root',        
    password=''
)

# 新增資料庫
def create_database(database_name):
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    cursor.close()

# 新增資料表
def create_table(database_name, table_name):
    cursor = connection.cursor()
    cursor.execute(f"USE {database_name}")
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (Time int(20) NOT NULL,Rain char(20) DEFAULT NULL,Inundation char(20) DEFAULT NULL,Rain_Rate float(20) DEFAULT NULL,Inundation_Rate float(20) DEFAULT NULL,Inundation_Area float(20) DEFAULT NULL,Inundation_Depth float(20) DEFAULT NULL,PRIMARY KEY (Time)) ENGINE=InnoDB DEFAULT CHARSET=utf8;")
    cursor.close()

# 新增資料
def insert_data(database_name, table_name, time, rain, inundation, rain_rate, inundation_rate, inundation_area, inundation_depth):
    cursor = connection.cursor()
    cursor.execute(f"USE {database_name}")
    cursor.execute(f"INSERT INTO {table_name} (Time, Rain, Inundation, Rain_Rate, Inundation_Rate, Inundation_Area, Inundation_Depth) VALUES (%s, %s, %s, %s, %s, %s, %s)", (time, rain, inundation, rain_rate, inundation_rate, inundation_area, inundation_depth))
    connection.commit()
    cursor.close()

# 關閉資料庫連線
def close_connection():
    if connection.is_connected():
        connection.close()
        print("資料庫連線已關閉")

if __name__ == "__main__":
    # 新增資料庫、資料表和資料
    create_database("UPEM")
    create_table("UPEM", "A2")
    insert_data("UPEM", "A2", 36000, "Yes", "Non", 92.03, 0, 0, 0)
    insert_data("UPEM", "A2", 37000, "Yes", "Non", 92.03, 0, 0, 0)
    insert_data("UPEM", "A2", 38000, "Yes", "Non", 92.03, 0, 0, 0)

    # 關閉資料庫連線
    close_connection()
