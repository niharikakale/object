from flask_wtf import FlaskForm
import cv2
from flask import Flask,render_template,request,redirect,send_file,url_for,Response,session
from werkzeug.utils import secure_filename,send_from_directory
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from wtforms.validators import InputRequired,NumberRange
import os
from video import video_detection
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
import bcrypt


app=Flask(__name__)
app.config['SECRET_KEY'] = 'gowthami'
app.config['UPLOAD_FOLDER'] = 'static\\uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all(),



@app.route('/',methods=["GET","POST"])
def index():
    session.clear()
    if request.method=="POST":
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']

            user = User.query.filter_by(email=email).first()

            if user and user.check_password(password):
                session['email'] = user.email
                return redirect('/page2')
            else:
                return render_template('Home page.html', error='Invalid user')
    return render_template("Home page.html")

@app.route('/sign-up',methods=["GET","POST"])
def page1():
    if request.method=="POST":
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/')
    return render_template("page 1.html")

@app.route('/page2')
def page2():
    return render_template("page2.html")



@app.route('/upload', methods=['GET','POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('upload_video.html', form=form)


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route("/predict_video",methods=["GET",'POST'])
def video():
    #return Response(generate_frames(path_x='video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/realtime_detection",methods=["GET",'POST'])
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__=='__main__':
    app.run(debug=True)
