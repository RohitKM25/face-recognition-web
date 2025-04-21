import face_recognition
import cv2
import face_recognition.face_recognition_cli
import numpy as np
import os
from datetime import datetime
import pickle as pkl
from settings import CONFIG, get_data_item_path

# from PIL import ImageFont, ImageDraw, Image
# INTER_FONT_S32 = ImageFont.truetype(
#     "Inter-VariableFont_slnt,wght.ttf", size=32)
# INTER_FONT_S32.set_variation_by_axes([700, 0])
# INTER_700_FONT_S32_HEIGHT = sum(np.abs(INTER_FONT_S32.getmetrics()))
# INTER_FONT_S24 = ImageFont.truetype(
#     "Inter-VariableFont_slnt,wght.ttf", size=24)
# INTER_FONT_S24.set_variation_by_axes([700, 0])
# INTER_700_FONT_S24_HEIGHT = sum(np.abs(INTER_FONT_S24.getmetrics()))

AVG_METHODS = {
    'mean': np.mean,
    'min': np.min
}

KNOWN_FACES_DATA = {}
KNOWN_FACES_NAMES = []
KNOWN_FACES_ENCODINGS = []
ABSENT_FACES_NAMES = []
RECOGNISED_FACES = []


def get_encoding_from_file(path):
    img = face_recognition.load_image_file(path)
    loc = face_recognition.face_locations(img, model=CONFIG['model_name'])
    return face_recognition.face_encodings(
        img, known_face_locations=loc, model="large",
        # num_jitters=100
    )[0]


def generate_known_faces_data(ignore_names=[]):
    data = {"names": [], "encodings": []}
    for person_folder_name in os.listdir(get_data_item_path('faces_images_folder_path')):
        if person_folder_name in ignore_names:
            continue
        person_folder_path = get_data_item_path(
            'faces_images_folder_path', person_folder_name)
        for person_image_name in os.listdir(person_folder_path):
            data['names'].append(person_folder_name)
            data['encodings'].append(get_encoding_from_file(
                os.path.join(person_folder_path, person_image_name)))
    return data


def check_new_known_faces(data):
    person_folders = os.listdir(get_data_item_path('faces_images_folder_path'))
    new_person_names = [
        person_name for person_name in person_folders if person_name not in data['names']]
    if len(new_person_names) == 0:
        return (False, None)
    return (True, generate_known_faces_data([i for i in data['names'] if i not in new_person_names]))


def check_updated_known_faces(data):
    person_folders = os.listdir(get_data_item_path('faces_images_folder_path'))
    updated_person_names = [
        person_name for person_name in person_folders if data['names'].count(person_name) < len(os.listdir(f'{get_data_item_path('faces_images_folder_path')}/{person_name}'))]
    if len(updated_person_names) == 0:
        return (False, None)
    return (True, generate_known_faces_data([i for i in data['names'] if i not in updated_person_names]))


def dump_known_faces_to_data_file(data):
    with open(get_data_item_path('faces_data_file_path'), mode="wb") as kfd_file:
        pkl.dump(data, kfd_file)


def load_known_faces_from_data_file():
    dt = None
    with open(get_data_item_path('faces_data_file_path'), mode="rb") as kfd_file:
        dt = pkl.load(kfd_file)
    return dt


def load_update_known_faces_data():
    KNOWN_FACES_DATA.update(load_known_faces_from_data_file())
    has_new_faces_data, new_faces_data = check_new_known_faces(
        KNOWN_FACES_DATA)
    if has_new_faces_data:
        for i in KNOWN_FACES_DATA:
            KNOWN_FACES_DATA[i].extend(new_faces_data[i])
        dump_known_faces_to_data_file(KNOWN_FACES_DATA)
    has_updated_faces_data, updated_faces_data = check_updated_known_faces(
        KNOWN_FACES_DATA)
    if has_updated_faces_data:
        for i in KNOWN_FACES_DATA:
            KNOWN_FACES_DATA[i].extend(updated_faces_data[i])
        dump_known_faces_to_data_file(KNOWN_FACES_DATA)
    print()


def create_known_faces_data():
    KNOWN_FACES_DATA.update(generate_known_faces_data())
    dump_known_faces_to_data_file(KNOWN_FACES_DATA)
    print()


def check_known_faces_data():
    if os.path.exists(get_data_item_path('faces_data_file_path')):
        load_update_known_faces_data()
    elif os.path.exists(get_data_item_path('faces_images_folder_path')):
        create_known_faces_data()
    else:
        return False
    KNOWN_FACES_NAMES.extend(KNOWN_FACES_DATA['names'])
    KNOWN_FACES_ENCODINGS.extend(KNOWN_FACES_DATA['encodings'])
    ABSENT_FACES_NAMES.extend(KNOWN_FACES_NAMES.copy())
    print()
    return True


# def detect_face_from_frame(frame):
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     face_encodings = face_recognition.face_encodings(
#         small_frame, model="small")
#     if len(face_encodings) == 0:
#         return (frame, None)
#     face_matches = face_recognition.compare_faces(KNOWN_FACES_ENCODINGS,
#                                                   face_encodings[0], CONFIG['maximum_tolerance'])
#     matched_names = [KNOWN_FACES_NAMES[i]
#                      for i in range(len(face_matches)) if face_matches[i]]
#     if len(matched_names) == 0:
#         return (frame, None)
#     matched_name = matched_names[0]

#     img_pil = Image.fromarray(frame)
#     img_draw = ImageDraw.ImageDraw(img_pil)
#     img_draw.text((20, INTER_700_FONT_S24_HEIGHT-INTER_700_FONT_S32_HEIGHT), "Detected",
#                   font=INTER_FONT_S24, fill=(255, 255, 255))
#     img_draw.text((20, INTER_700_FONT_S32_HEIGHT), matched_name,
#                   font=INTER_FONT_S32, fill=(255, 255, 255))

#     return (np.array(img_pil), (matched_name, datetime.now().strftime("%H:%M:%S")))


def detect_face_from_frame(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_encodings = face_recognition.face_encodings(
        small_frame, model="small")
    if len(face_encodings) == 0:
        return None
    face_matches = face_recognition.compare_faces(KNOWN_FACES_ENCODINGS,
                                                  face_encodings[0], CONFIG['maximum_tolerance'])
    matched_names = [KNOWN_FACES_NAMES[i]
                     for i in range(len(face_matches)) if face_matches[i]]
    if len(matched_names) == 0:
        return None
    matched_name = matched_names[0]

    return matched_name, datetime.now().strftime("%H:%M:%S")
