from pickle import TRUE
import secrets
from flask import Flask,request, url_for, redirect, render_template,flash,current_app
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from flask_mail import Message
from flask_login import UserMixin
from flask_login import login_user, current_user, logout_user, login_required
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
import pygal
import urllib.request
import argparse
import os
from typing import Tuple, List
from werkzeug.utils import secure_filename
from pprint import pprint
import cv2
from PIL import Image
import numpy as np
import re
import pytesseract
from flaskApp.segmentation import detectionPage, detectionWord, sort_words
from flaskApp.model import Model, DecoderType
from flaskApp.preprocessor import Preprocessor
from flaskApp.dataloader_iam import Batch
import difflib
from spellchecker import SpellChecker
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib.pyplot import plot
import torch, torchvision
from skimage import util 
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tensorflow import keras
import tensorflow as tf
from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

app = Flask(__name__)



ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'tif']

UPLOAD_FOLDER = 'static/uploads'
AXIS_BANK = 'D:/SimpleHTR/static/cheque/axis/'
CANARA_BANK = 'D:/SimpleHTR/static/cheque/canara/'
ICICI_BANK = 'D:/SimpleHTR/static/cheque/icici/'
SYNDICATE_BANK = 'D:/SimpleHTR/static/cheque/syndicate/'
LETTRE = 'D:/SimpleHTR/static/cheque/lettre/'
CHIFFRE = 'D:/SimpleHTR/static/cheque/segments_chiffre'
SEGMENTATION = 'D:/SimpleHTR/static/cheque/segments'
STATIC ='D:/SimpleHTR/static'
BLANK_IMAGE = 'D:/SimpleHTR/static/cheque/blank.png'
LEGAL='D:/SimpleHTR/static/legal.png'
COURTESY='D:/SimpleHTR/static/courtesy.png'

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
mail = Mail()

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
Montant_chiffre = 0

class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Update')

    def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('That email is taken. Please choose a different one.')


class RequestResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = 'D:/SimpleHTR/model/charList.txt'
    fn_number_list = 'D:/SimpleHTR/model/NumberList.txt'
    fn_summary = 'D:/SimpleHTR/model/summary.json'
    fn_corpus = 'D:/SimpleHTR/data/corpus.txt'

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('users.reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)

def preprocessing_before_crop(image_path):
    img_path = image_path
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    scale_percent = 100  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_colred = cv2.cvtColor(cv2.resize(image, dim), cv2.COLOR_BGR2GRAY)
    image_filtred1 = cv2.bilateralFilter(image_colred, 5, 75, 75)
    image_threshed = cv2.adaptiveThreshold(image_filtred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 221,
                                           12)
    image_blured = cv2.medianBlur(image_threshed,
                                  3)  # 3 - 5 is the best value so far because the 7 make the 'c' of lalch (2nd word not clear)
    cropped_image = image_blured[40:530, 200:2285]
    image_bordered = cv2.copyMakeBorder(cropped_image, 3, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image_bordered

def get_contour_precedence_amount(contour_amount, cols_amount):
    tolerance_factor_amount = 100
    origin_amount = cv2.boundingRect(contour_amount)
    return ((origin_amount[1] // tolerance_factor_amount) * tolerance_factor_amount) * cols_amount + origin_amount[0]


def amount_cropping(cropped_image):
    amount_image = cropped_image[300:490, 1450:2085]
    v_amount_image = np.median(amount_image)
    sigma_amount_image = 0.33
    lower_amount_image = int(max(0, (1.0 - sigma_amount_image) * v_amount_image))
    upper_amount_image = int(min(255, (1.0 + sigma_amount_image) * v_amount_image))
    canned_amount_image = cv2.Canny(amount_image, upper_amount_image, lower_amount_image)
    lines_amount_image = cv2.HoughLinesP(canned_amount_image, 1, np.pi / 180, 300, minLineLength=300, maxLineGap=600)
    for line in lines_amount_image:
        x1, y1, x2, y2 = line[0]
        cv2.line(amount_image, (x1, y1), (x2, y2), (255, 0, 0), 9)
    amount_image_copy = amount_image.copy()
    kernel = np.ones((1, 1), np.uint8)
    amount_image_copy_erosion = cv2.erode(amount_image_copy, kernel, iterations=3)
    contours_amount, hierarchy_amount = cv2.findContours(image=amount_image_copy_erosion, mode=cv2.RETR_TREE,
                                                         method=cv2.CHAIN_APPROX_SIMPLE)
    contours_amount = list(contours_amount)
    contours_amount.sort(key=lambda x: get_contour_precedence_amount(x, amount_image_copy_erosion.shape[1]))
    ROI_number_amount = 0
    for c in contours_amount:
        x, y, w, h = cv2.boundingRect(c)
        if 30 < h < 85 and 2 < w < 85:
            cv2.rectangle(amount_image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            ROI_amount = amount_image[y:y + h, x:x + w]
            cv2.imwrite(r'D:/SimpleHTR/static/cheque/segments_chiffre/segment_{}.png'.format(ROI_number_amount), ROI_amount)
            ROI_number_amount += 1

    return amount_image

def deleteSegmentChiffre():
    base_path = 'D:/SimpleHTR-master/static/cheque/segments_chiffre/'
    for infile in os.listdir(base_path):
        if infile.endswith('png') and infile.startswith('segment'):
            os.remove(base_path+infile)

def Delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    return True

def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img



def recognitionMontantEnChiffre():
    segment_chiffre='D:/SimpleHTR/static/cheque/segments_chiffre/'
    model = tf.keras.models.load_model('D:/SimpleHTR/static/public/digits_recognition.h5')
    directory=os.listdir(segment_chiffre)
    directory.sort()
    montant_chiffre=[]
    dict_montant={}
    for segment in directory:
        print(segment)
        if not "ipynb_checkpoints" in segment:
            path = segment_chiffre + segment
            print(path)
            if segment !='segment_0.png':
                seg=load_image(path)
                seg = np.expand_dims(seg, axis=0)
                pred=model.predict(seg)
                pred_idx = np.argmax(pred)
                montant_chiffre.append(pred_idx)
                dict_montant[pred_idx]='1'
    return montant_chiffre,dict_montant

def center_crop(im, crop_pixels=45):
  return im[crop_pixels:im.shape[0] - crop_pixels * 3, crop_pixels:im.shape[1] - crop_pixels]

def bright_contrast_loop(image, alpha=1, beta=0):
  new_image = np.zeros(image.shape, image.dtype)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      for c in range(image.shape[2]):
        new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
  return new_image

def readingMontantChiffre(montant_chif):
  for i in range(len(montant_chif)):
    montant_chif[i]=str(montant_chif[i])
  m=''.join(montant_chif)
  print(m)
  Montant_chiffre = m

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS




@app.route("/dachboard")
def dashboard():
    
    bar_chart = pygal.HorizontalBar()
    bar_chart.title = 'Number of checks by amount (in thousand checks)'
    bar_chart.add('<20 TND', 180)
    bar_chart.add('100 <> 20 TND', 4000)
    bar_chart.render()
    bar_chart_tn = bar_chart.render_data_uri()

    graph_tn = pygal.Line()
    graph_tn.title = 'VOLUME OF TRANSACTIONS BY METHODS OF PAYYMENT (IN MDT)'
    graph_tn.x_labels = ['MaY','June','July','Aug','Sept','Oct']
    graph_tn.add('Checks',  [5067, 7815, 8226, 7105, 8855, 8363])
    graph_tn.add('Effects',    [1514, 2205, 1604.0, 2114, 2068, 1807])
    graph_tn.add('Wire Transfers',     [2681,  3085, 3492, 2773, 3224, 3319])
    graph_tn.add('Debit Credit',     [275,  2194, 1451, 276, 2547, 630])
    graph_tn.add('Credit Cards',     [1231,  1418, 1558, 1482, 1355])
    graph_tn.add('E-Pay',     [20,  24, 31, 25, 23])

    graph_data_tn = graph_tn.render_data_uri()

    graph = pygal.Line()
    graph.title = 'Trends in noncash payments, by number.'
    graph.x_labels = ['2000','2003','2006','2009','2012','2015','2018']
    graph.add('Checks',  [42, 39, 30, 27, 21, 19, 15])
    graph.add('Debit cards',    [9, 18, 25, 40,  48,  53, 72])
    graph.add('ACH Credit Transfer',     [2,  4, 8, 9, 14, 16, 18])
    graph_data = graph.render_data_uri()

    graph2 = pygal.Line()
    graph2.title = 'Trends in noncash payments, by value.'
    graph2.x_labels = ['2000','2003','2006','2009','2012','2015','2018']
    graph2.add('Checks',  [40, 41, 42, 34, 28, 31, 25])
    graph2.add('Debit cards',    [1, 2, 3, 4,  5,  6, 7])
    graph2.add('ACH Credit transfer',     [9,  12, 18, 21, 28, 32, 41])
    graph_data2 = graph2.render_data_uri()

    return render_template('dashboard.html', title='Dashboard', bar_chart_tn=bar_chart_tn, graph_data=graph_data, graph_data2=graph_data2, graph_data_tn=graph_data_tn)
@app.route("/")
@app.route("/homepage")
def homepage():
    return render_template('home.html', title='homepage')

@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)

@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)

@app.route("/check-autocorrection")
def checkAuto():
    return render_template('checkAuto.html', title='CheckAuto')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('users.login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/predict", methods=['POST'])
def prediction():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return redirect('/converting')
        #return redirect(url_for('cheque.show_image'))
    #elif request.method == 'GET':
     #   form1 = ChequeCorretionForm()
      #  filename = send_from_directory(UPLOAD_FOLDER,
       #                        filename, as_attachment=True)
    #if file.filename == '':
     #   flash('no image selected')
      #  return redirect('/check-autocorrection')

    #if file and allowed_file(file.filename):
      #  filename = secure_filename(file.filename)
      #  file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('login.html', title='Login')
    #else:
        #flash('Allowed types are jpg jpeg png gif only !')
        #return redirect('/converting')




@app.route("/converting", methods=['GET'])
def converting():
    for infile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in infile:
            read = cv2.imread(UPLOAD_FOLDER + '/' + infile)
            outfile = infile.split('.')[0] + '.png'
            cv2.imwrite(UPLOAD_FOLDER + '/' + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
            os.remove(UPLOAD_FOLDER + '/'+infile)
            return redirect('/cropcourtesyamount')

@app.route("/cropcourtesyamount", methods=['GET'])
def cropcourtesyamount():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\admin\AppData\Local\Tesseract-OCR\tesseract.exe'
    for file in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in file:
            image = cv2.imread(UPLOAD_FOLDER + '/' + file)
            extract = pytesseract.image_to_string(Image.open(UPLOAD_FOLDER + '/' + file))
            extract = ''.join(extract)
            extract = extract.split(' ')
            l = list()
            for i in extract:
                if i != '' and i != ' ':
                    l.append(i)
            print(l)
            for i in l:
                if re.search("^axi.*", i.lower()):
                    courtesy = image[300:490, 1450:2085]
                    cv2.imwrite('D:/SimpleHTR/static/courtesyamt.png', courtesy)
                elif re.search("^syn.*", i.lower()) or re.search("^dicat.*", i.lower()):
                    courtesy = image[400:550, 1650:2085]
                    cv2.imwrite('D:/SimpleHTR/static/courtesyamt.png', courtesy)
                elif re.search("^icic.*", i.lower()):
                    courtesy = image[400:500, 1860:2300]
                    cv2.imwrite('D:/SimpleHTR/static/courtesyamt.png', courtesy)
                elif re.search("^cana.*", i.lower()):
                    courtesy = image[440:540, 1840:2280]
                    cv2.imwrite('D:/SimpleHTR/static/courtesyamt.png', courtesy)
            return redirect('/croplegalamount')

@app.route("/croplegalamount", methods=['GET'])
def croplegalamount():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\admin\AppData\Local\Tesseract-OCR\tesseract.exe'
    for file in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in file:
            image = cv2.imread(UPLOAD_FOLDER + '/' + file)
            extract = pytesseract.image_to_string(Image.open(UPLOAD_FOLDER + '/' + file))
            extract = ''.join(extract)
            extract = extract.split(' ')
            l = list()
            for i in extract:
                if i != '' and i != ' ':
                    l.append(i)
            print(l)
            for i in l:
                if re.search("^axi.*", i.lower()):
                    legal = image[320:440, 220:1720]
                    cv2.imwrite('D:/SimpleHTR/static/legal.png', legal)
                elif re.search("^syn.*", i.lower()) or re.search("^dicat.*", i.lower()):
                    legal = image[310:440, 350:1830]
                    cv2.imwrite('D:/SimpleHTR/static/legal.png', legal)
                elif re.search("^icic.*", i.lower()):
                    legal = image[310:430, 450:2250]
                    cv2.imwrite('D:/SimpleHTR/static/legal.png', legal)
                elif re.search("^cana.*", i.lower()):
                    legal = image[320:440, 350:1850]
                    cv2.imwrite('D:/SimpleHTR/static/legal.png', legal)
            return redirect('/cropChiffre')

@app.route("/cropChiffre", methods=['GET','POST'])
def cropChiffre():
    for infile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in infile:
            path_to_image = UPLOAD_FOLDER + '/' + infile
            cropped_image = preprocessing_before_crop(path_to_image)
            amount_image = amount_cropping(cropped_image)
            return redirect('/preprocessing')

@app.route("/preprocessing", methods=['GET'])
def preprocessing():
    for outfile in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in outfile:
            image = cv2.imread(UPLOAD_FOLDER + '/' + outfile)
            resized = cv2.resize(image, (2300, 1000), )

            cropped = center_crop(resized)

            contrast_im = bright_contrast_loop(cropped, alpha=1.07)

            graychiffre = cv2.cvtColor(contrast_im, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(graychiffre, 150, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            result = 255 - opening

            #dst = cv2.fastNlMeansDenoisingColored(result, None, 30, 30, 21, 45)

            kernel = np.ones((2, 2), np.uint8)
            erosion = cv2.erode(result, kernel, iterations=1)

            cv2.imwrite(UPLOAD_FOLDER + '/' + outfile, erosion)
            return redirect('/repartir')

@app.route("/repartir", methods=['GET'])
def repartition():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\admin\AppData\Local\Tesseract-OCR\tesseract.exe'
    for file in os.listdir(UPLOAD_FOLDER):
        if not "ipynb_checkpoints" in file:
            extract = pytesseract.image_to_string(Image.open(UPLOAD_FOLDER + '/' + file))
            image = cv2.imread(UPLOAD_FOLDER + '/' + file)
            extract = ''.join(extract)
            extract = extract.split(' ')
            l = list()
            for i in extract:
                if i != '' and i != ' ':
                    l.append(i)
            print(l)
            for i in l:
                if re.search("^axi.*", i.lower()):
                    cv2.imwrite(AXIS_BANK + file, image)
                elif re.search("^synd.*", i.lower()) or re.search("^dicat.*", i.lower()):
                    cv2.imwrite(SYNDICATE_BANK + file, image)
                elif re.search("^icic.*", i.lower()):
                    cv2.imwrite(ICICI_BANK + file, image)
                elif re.search("^cana.*", i.lower()):
                    cv2.imwrite(CANARA_BANK + file, image)
            return redirect('/extract')

@app.route("/extract", methods=['GET'])
def extraction():
    for infile_axis in os.listdir(AXIS_BANK):
        if not "ipynb_checkpoints" in infile_axis:
            y_axis = 245
            x_axis = 200
            h_axis = 86
            w_axis = 2000
            im_axis = cv2.imread(AXIS_BANK + infile_axis)
            crop_axis = im_axis[y_axis:y_axis + h_axis, x_axis:x_axis + w_axis]
            cv2.imwrite(LETTRE + infile_axis, crop_axis)

            im_axis = Image.open(r'D:/SimpleHTR/static/cheque/blank.png')
            blank_axis = im_axis.copy()

            crop_axis_2 = Image.open(r'D:/SimpleHTR/static/cheque/lettre/' + infile_axis)
            blank_axis.paste(crop_axis_2, (90, 300))
            blank_axis.save(r'D:/SimpleHTR/static/cheque/lettre/' + infile_axis)

    for infile_canara in os.listdir(CANARA_BANK):
        if not "ipynb_checkpoints" in infile_canara:
            y_canara = 238  # 255
            x_canara = 300
            h_canara = 84
            w_canara = 2000
            im_canara = cv2.imread(CANARA_BANK + infile_canara)
            crop_canara = im_canara[y_canara:y_canara + h_canara, x_canara:x_canara + w_canara]
            cv2.imwrite(LETTRE + infile_canara, crop_canara)

            im_canara = Image.open(r'D:/SimpleHTR/static/cheque/blank.png')
            blank_canara = im_canara.copy()

            crop_canara_2 = Image.open(r'D:/SimpleHTR/static/cheque/lettre/' + infile_canara)
            blank_canara.paste(crop_canara_2, (90, 300))
            blank_canara.save(r'D:/SimpleHTR/static/cheque/lettre/' + infile_canara)

    for infile_icici in os.listdir(ICICI_BANK):
        if not "ipynb_checkpoints" in infile_icici:
            y_icici = 240
            x_icici = 300
            h_icici = 83
            w_icici = 2000
            im_icici = cv2.imread(ICICI_BANK + infile_icici)
            crop_icici = im_icici[y_icici:y_icici + h_icici, x_icici:x_icici + w_icici]
            cv2.imwrite(LETTRE + infile_icici, crop_icici)

            im_icici = Image.open(r'D:/SimpleHTR/static/cheque/blank.png')
            blank_icici = im_icici.copy()

            crop_icici_2 = Image.open(r'D:/SimpleHTR/static/cheque/lettre/' + infile_icici)
            blank_icici.paste(crop_icici_2, (90, 300))
            blank_icici.save(r'D:/SimpleHTR/static/cheque/lettre/' + infile_icici)

    for infile_syndicate in os.listdir(SYNDICATE_BANK):
        if not "ipynb_checkpoints" in infile_syndicate:
            y_syndicate = 258
            x_syndicate = 300
            h_syndicate = 83
            w_syndicate = 2000
            im_syndicate = cv2.imread(SYNDICATE_BANK + infile_syndicate)
            crop_syndicate = im_syndicate[y_syndicate:y_syndicate + h_syndicate, x_syndicate:x_syndicate + w_syndicate]
            cv2.imwrite(LETTRE + infile_syndicate, crop_syndicate)

            im_syndicate = Image.open(r'D:/SimpleHTR/static/cheque/blank.png')
            blank_syndicate = im_syndicate.copy()

            crop_syndicate_2 = Image.open(r'D:/SimpleHTR/static/cheque/lettre/' + infile_syndicate)
            blank_syndicate.paste(crop_syndicate_2, (90, 300))
            blank_syndicate.save(r'D:/SimpleHTR/static/cheque/lettre/' + infile_syndicate)
    return redirect('/recognition')

def segmentationImage(img_name: str) -> None:
    image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    # Crop image and get bounding boxes r'C:\Users\yassi\Desktop\segment.png'
    assert image is not None
    crop = detectionPage(image)
    boxes = detectionWord(crop)
    lines = sort_words(boxes)

    # Saving the bounded words from the page image in sorted way
    i = 0
    for line in lines:
        text = crop.copy()
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            save = Image.fromarray(text[y1:y2, x1:x2])
            # print(i)
            save.save(r'D:/SimpleHTR/static/cheque/segments/segment' + str(i) + '.png')
            i += 1

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def deleteSegment():
    base_path = 'D:/SimpleHTR/static/cheque/segments/'
    for infile in os.listdir(base_path):
        if infile.endswith('png') and infile.startswith('segment'):
            os.remove(base_path+infile)

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def SpellingMistakeCorrection(initial_data:dict):
    all_data = {}
    list_of_words = ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","hundred","thousand","million","lakh","lakhs"]
    list_of_special_car = ['*', ',', '.', '/', '#', '+', '-', ';', ':', '?', '!', '%', '^', '~', '{', '}', '[', ']', '(',')', '_', '@']
    spell = SpellChecker(language=None)
    spell.word_frequency.load_text_file(r'D:/SimpleHTR/data/my_text_file.txt')
    data = []
    probs = []
    for i, j in initial_data.items():
        if '0' not in i and '1' not in i and '2' not in i and '3' not in i and '4' not in i and '5' not in i and '6' not in i and '7' not in i and '8' not in i and '9' not in i:
            all_data[i] = j
    for i in range(len(list_of_special_car)):
        if list_of_special_car[i] in all_data:
            del all_data[list_of_special_car[i]]
    for i, j in all_data.items():
        data.append(i)
        probs.append(j)

    data = [x.lower() for x in data]

    for a in range(len(data)):
        data[a]=spell.correction(data[a])

    for i in range(len(data)):
        list_ratios=[]
        if data[i] not in list_of_words:
            string_incorr = data[i]
            for j in list_of_words:
                string = j
                emp = difflib.SequenceMatcher(None,string_incorr,string)
                list_ratios.append(emp.ratio())
                max_value = max(list_ratios)
                index = list_ratios.index(max_value)
                data[i]=list_of_words[index]
    final_dict= dict(zip(data, probs))
    return final_dict

def infer(model: Model, fn_img: str):
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)

    return recognized, probability
def recognitionMontantEnLettre(model: Model):
    base_path = 'D:/SimpleHTR/static/cheque/segments/'
    data={}
    print("Scanning ...")
    for infile in os.listdir(base_path):
        if infile.endswith('png'):
            r, p = infer(model, base_path+infile)
            data[r[0]] = p[0]
    print('recongnized : ', data)
    a=SpellingMistakeCorrection(data)
    print('corrected : ', a)
    return a


def num2words(num):
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
                'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand', 100000: 'Lakhs', 10000000: 'Crores'}
    if num < 20:
        return under_20[num]
    if num < 100:
        return tens[num // 10 - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])
    pivot = max([key for key in above_100.keys() if key <= num])

    return num2words(num // pivot) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))

def comparaisonMontant(montant_chiff,montant_lett:dict):
  montant_chiffre=num2words(montant_chiff)
  montant_chiffre=montant_chiffre.lower()
  c=[]
  for key,value in montant_lett.items():
    c.append(key)
  montant_lettre=' '.join(c)
  montant_lettre=montant_lettre.lower()
  if 'lakhs' in montant_chiffre:
    montant_chiffre=montant_chiffre.replace('lakhs','lakh')

  if montant_chiffre == montant_lettre :
    print('cheque correcte')
    return "Verified"
  else:
    print('cheque à verifier')
    return "Autocorrected"

@app.route("/recognition", methods=['GET'])
def recognition():

    
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}

    decoder_type = decoder_mapping['wordbeamsearch']
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump='store_true')


    for lettre in os.listdir(LETTRE):
        if not "ipynb_checkpoints" in lettre:
            segmentationImage(r'D:/SimpleHTR/static/cheque/lettre/' + lettre)
            montant_lett = recognitionMontantEnLettre(model)
            deleteSegment()
    c = []
    segment_chiffre='D:/SimpleHTR/static/cheque/segments_chiffre/'
    mnist_model = tf.keras.models.load_model('D:/SimpleHTR/static/public/digits_recognition.h5')
    directory=os.listdir(segment_chiffre)
    directory.sort()
    montant_chiffre=[]
    dict_montant={}
    Montant_chiffre=0
    mnist_pred=""
    for segment in directory:
        print(segment)
        if not "ipynb_checkpoints" in segment:
            path = segment_chiffre + segment
            print(path)
            if segment !='segment_0.png':
                seg=load_image(path)
                seg = np.expand_dims(seg, axis=0)
                pred=mnist_model.predict(seg)
                pred_idx = np.argmax(pred)
                mnist_pred=mnist_pred + str(pred.argmax())
                montant_chiffre.append(pred_idx)
                dict_montant[pred_idx]='1'
    s = [str(integer) for integer in montant_chiffre]
    Montant_chiffre = "".join(s)

    Montant_chiffre = int(Montant_chiffre)
    #Montant_chiffre=readingMontantChiffre(montant_chiffre)
 
    for key, value in montant_lett.items():
        c.append(key)
    montant_lettre = ' '.join(c)
    montant_lettre = montant_lettre.lower()
    flash(montant_lettre)
    flash(Montant_chiffre)
    #montant_lettre_chiffre = num2words(Montant_chiffre)
    a=comparaisonMontant(Montant_chiffre,montant_lett)
    Delete_all_files(UPLOAD_FOLDER)
    Delete_all_files(AXIS_BANK)
    Delete_all_files(CANARA_BANK)
    Delete_all_files(ICICI_BANK)
    Delete_all_files(SYNDICATE_BANK)
    Delete_all_files(LETTRE)
    Delete_all_files(CHIFFRE)
    Delete_all_files(SEGMENTATION)
    return render_template('checkVerif.html',comparaison=a, chiffre = mnist_pred,lettre=montant_lettre)

@app.route("/resultat", methods=['GET','POST'])
def resultat():
    
    return render_template('checkVerif.html', title='CheckVerif')

if __name__ == '__main__':
    app.run(debug=True)