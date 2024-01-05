import os
import time
from pathlib import Path


def convert_jpg_to_png(directory_name):
    """
    Converts all JPG files in the specified directory to PNG files with their names
    as Unix timestamps.

    :param directory_name: The name of the directory containing JPG files.
    """
    directory = Path(directory_name)
    for file_path in directory.glob('*.jpg'):
        timestamp = convert_time_to_timestamp(file_path.stem)
        new_filename = f"{timestamp}.png"
        new_file_path = file_path.with_name(new_filename)
        file_path.rename(new_file_path)
        print(f"{file_path.stem} -> {timestamp}")


def convert_time_to_timestamp(time_string):
    """
    Converts a time string in the format 'YYYY-MM-DD_HHhMM' to a Unix timestamp.

    :param time_string: The time string to convert.
    :return: The Unix timestamp as an integer.
    """
    struct_time = time.strptime(time_string, "%Y-%m-%d_%Hh%M")
    return int(time.mktime(struct_time))


def convert_timestamp_to_time(timestamp):
    """
    Converts a Unix timestamp to a time string in the format 'YYYY-MM-DD_HHhMM'.

    :param timestamp: The Unix timestamp to convert.
    :return: The formatted time string.
    """
    local_time = time.localtime(int(timestamp))
    return time.strftime("%Y-%m-%d_%Hh%M", local_time)


if __name__ == "__main__":
    # Example usage:
    # directory_name = input("Enter the directory name: ")
    # convert_jpg_to_png(directory_name)

    # Example conversion from time string to timestamp
    # time_string = "2020-08-10_00h30"
    # print(convert_time_to_timestamp(time_string))

    # Example conversion from timestamp to time string
    timestamp = 1596990600
    print(convert_timestamp_to_time(timestamp))