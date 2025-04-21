import os
from datetime import datetime
from settings import CONFIG, get_data_item_path
import csv


def get_current_attendance_file_name(): return get_data_item_path('attendance_folder_path', f'{
    datetime.now().strftime("%Y-%m-%d")}.csv')


def load_attendance_data():
    attendance_file_name = get_current_attendance_file_name()
    file_exists = os.path.exists(attendance_file_name)
    attendance_data = []
    if file_exists:
        with open(attendance_file_name, 'r', newline='') as attendance_data_file:
            lnreader = csv.reader(attendance_data_file)
            rows = [r for r in lnreader]
            attendance_data.extend(rows)
    return attendance_data


def add_person(person_data):
    if person_data[0] in [i[0] for i in load_attendance_data()]:
        return False
    attendance_file_name = get_current_attendance_file_name()
    with open(attendance_file_name, 'a' if os.path.exists(attendance_file_name) else 'w', newline='') as attendance_data_file:
        lnwriter = csv.writer(attendance_data_file)
        lnwriter.writerow(person_data)
    return True
