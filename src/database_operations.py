from packages import database


def create_database(ville, commener):
    database_name = ville + "_" + commener
    database.create_database(database_name)
    return database_name

def store_data(database_name, time_start, rain_class, inundation_class, car_detection, water_detection, ground_detection, crosswalk_detection, inundated_region, inundation_depth_val):
    database.store_data(database_name, time_start, rain_class, inundation_class, car_detection, water_detection, ground_detection, crosswalk_detection, inundated_region, inundation_depth_val)
