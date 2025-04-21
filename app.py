import cv2
import os
from flask import Flask, flash, request, redirect, url_for, stream_with_context, render_template, Response
from werkzeug.utils import secure_filename
from settings import check_settings
from facerecognition import detect_face_from_frame, check_known_faces_data
from attendance import add_person

check_settings()
check_known_faces_data()
app = Flask(__name__)

DETECTED_FACES = [['fwef']]


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv


def generate_processed_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture frame")
            break
        person = detect_face_from_frame(frame)
        if person:
            add_person(person)
        DETECTED_FACES.append(person)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/new/person', methods=['GET', 'POST'])
def generate_encodings():
    if request.method == 'POST':
        person_name = request.form['name']
        person_folder_path = f'data/photos/{person_name}'
        for img_file in request.files.getlist('img-file'):
            if not os.path.exists(person_folder_path):
                os.mkdir(person_folder_path)
            img_file.save(
                f'{person_folder_path}/{secure_filename(img_file.filename)}')
    return render_template('new-person.html', post_message="Successfully Uploaded Images" if request.method == 'POST' else "")


if __name__ == '__main__':
    app.run(debug=True)
