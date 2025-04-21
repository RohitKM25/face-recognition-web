import json
import os

CONFIG = {
    'data_folder_path': 'data',
    'faces_images_folder_path': 'photos',
    'attendance_folder_path': 'attendance',
    'faces_data_file_path': 'known_faces.bin',
    'model_name': 'hog',
    'maximum_tolerance': 0.45,
    'avg_method': 'mean',
    'output_verbose': True,
}


def dump_settings_file():
    with open("settings.json", mode='w') as settings_file:
        json.dump(CONFIG, settings_file)


def check_data_folder():
    for i in CONFIG:
        if 'folder_path' not in i:
            continue
        dp = get_data_item_path(i)
        if os.path.exists(dp):
            continue
        os.mkdir(dp)


def check_settings():
    if os.path.exists("settings.json"):
        with open("settings.json", mode='r') as settings_file:
            CONFIG.update(json.load(settings_file))
    else:
        dump_settings_file()
    check_data_folder()


def get_data_item_path(key, *paths):
    return CONFIG['data_folder_path'] if "data_folder_path" == key else os.path.join(CONFIG['data_folder_path'], CONFIG[key], *paths)


def set_settings(key, value):
    if key in CONFIG:
        CONFIG[key] = value
