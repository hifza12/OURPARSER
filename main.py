from flask import Flask, render_template, json,request, redirect, url_for, session, flash, make_response
#from flask_mysqldb import MySQL
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.facebook import make_facebook_blueprint, facebook
import emailparser as ep
import text_parsing as tp
#import new_text as tp
app= Flask(__name__)
app.secret_key="Secret Key"

app.config['SQLALCHEMY_DATABASE_URI']="mysql+pymysql://root:@localhost/EmailParser"
# app.config['SQLALCHEMY_TRACK_NOTIFICATIONS']=False
db = SQLAlchemy(app)

@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return render_template('index.html')
    else:
        return render_template('index.html', message="Hello!")

class Register(db.Model):
    #user_id,user_fullname,user_phno,user_email,user_message
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(80), nullable=False)
    user_password=db.Column(db.String(128),nullable=False)
    user_email=db.Column(db.String(80),nullable=False)

    def __init__(self,user_name,user_password,user_email):
        self.user_name=user_name
        self.user_password=generate_password_hash(user_password, method='sha256')
        self.user_email=user_email

class Login(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(80), nullable=False)
    user_password = db.Column(db.String(128), nullable=False)

    def __init__(self, user_email, user_password):
        self.user_email = user_email
        self.user_password = generate_password_hash(user_password, method='sha256')

class LoginForm():
    def __init__(self):
        self.user_email = ''
        self.user_password = ''


@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'user_name' in request.form and 'user_password' in request.form and 'user_email' in request.form:
        # Create variables for easy access
        user_name = request.form['user_name']
        user_password = request.form['user_password']
        user_email = request.form['user_email']
        user = Register.query.filter_by(user_name=user_name).first()
        if user:
            return 'Username already exists!'
        new_user = Register(user_name=user_name, user_password=user_password, user_email=user_email)
        db.session.add(new_user)
        db.session.commit()
        session['user_name'] = user_name
        return redirect(url_for('login'))
        # return redirect(url_for('index.html'))
    # return render_template('register.html')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
    # return "user not entered"


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        user_email = request.form['user_email']
        user_password = request.form['user_password']
        data = Register.query.filter_by(user_email=user_email, user_password= check_password_hash(user_password, user_password)).first()
        if data is not None:
            session['logged_in'] = True
            return redirect(url_for("home"))
            #return "login successfully"
           # return redirect(url_for('tables'))

           # return r edirect(url_for('index'),message="Login In sucessfully")
        #return render_template('index.html', message="Incorrect Details")
        return "login fail"




@app.route('/forgot')
def forgot():
    return render_template('forgot-password.html')

@app.route('/text_parser',methods=['POST'])
def text_parser():
    if request.method == "POST":
        keywords = request.form['keywords']
        regex = request.form['regex']
        text = request.form['text']
        stopwords = request.form['proximity_stop_words']
        limit = request.form['limit']
        exactmatch = request.form['exactmatch']
        duplicates = request.form['duplicates']
        direction = request.form['direction']
       # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
        extracted_data = tp.document_extraction(keywords, regex,text,stopwords, limit,exactmatch, duplicates, direction)
        return render_template('utilities-other.html', res=extracted_data)
    return render_template('documentparser.html')


# @app.route('/email', methods=['POST'])
# def email_parser():
#     if request.method == "POST":
#         # (user: str, password: str, msg_from: str, value: str, keyword: str, regex: str,
#         #                      proximity_stop_words: str, limit, exact_match: bool, duplicates: bool, direction: str):
#             users=request.form['user']
#             passwords=request.form['password']
#             msg_from=request.form['msg_from']
#             value=request.form['value']
#             keywords = request.form['keyword']
#             regex = request.form['regex']
#             stopwords = request.form['proximity_stop_words']
#             limit = request.form['limit']
#             exactmatch = request.form['exact_match']
#             duplicates = request.form['duplicates']
#             direction = request.form['direction']
#             # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
#             extracted_data = ep.email_extraction(users,passwords,msg_from,value,keywords, regex, stopwords, limit, exactmatch, duplicates,
#                                                     direction)
#     # return render_template('utilities-other.html', res=extracted_data)
#             return extracted_data
#
#         # return extracted_data
#        # return render_template('documentparser.html')
#     return render_template('emailparser.html')

@app.route('/utilities')
def utlities():
    return render_template('utilities-other.html')

@app.route('/about_document')
def about_document():
    return render_template('about_document.html')
@app.route('/about_email')
def about_email():
    return render_template('about_email.html')

#callback
# @app.route('/utilities')
# def animation():
#     return render_template('utilities-other.html')

# @app.route('/document_parser')
# def document_parser():
#     return render_template('documentparser.html')

@app.route('/error_page')
def error():
    return render_template('404.html')

@app.route('/check_document', methods=['GET','POST'])
def check_document():
    if request.method == "POST":
        keywords = request.form['keywords']
        regex = request.form['regex']
        text = request.form['text']
        stopwords = request.form['proximity_stop_words']
        limit = request.form['limit']
        exactmatch = request.form['exactmatch']
        duplicates = request.form['duplicates']
        direction = request.form['direction']
       # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
        extracted_data = tp.document_extraction(keywords, regex,text,stopwords, limit,exactmatch, duplicates, direction)
        return render_template('utilities-other.html', res=extracted_data)
    return render_template('documentparser.html')



@app.route('/check_email', methods=['GET','POST'])
def check_email():
    if request.method == "POST":
        # (user: str, password: str, msg_from: str, value: str, keyword: str, regex: str,
        #                      proximity_stop_words: str, limit, exact_match: bool, duplicates: bool, direction: str):
        users = request.form['user']
        passwords = request.form['password']  # wyashvhufbssddga
        msg_from = request.form['msg_from']
        value = request.form['value']
        keywords = request.form['keyword']
        regex = request.form['regex']
        stopwords = request.form['proximity_stop_words']
        limit = request.form['limit']
        exactmatch = request.form['exact_match']
        duplicates = request.form['duplicates']
        direction = request.form['direction']
        # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
        extracted_data = ep.email_extraction(users, passwords, msg_from, value, keywords, regex, stopwords, limit,
                                             exactmatch, duplicates,
                                            direction)
        # return extracted_data
        if extracted_data:
            return render_template('charts.html', result=extracted_data)
        else:
            message = "No entities found in the email"
            return render_template('charts.html', message=message)

        return render_template('emailparser.html')

@app.route('/')
def charts():
    return render_template('index2.html')





if __name__=="__main__":
    app.run(debug=True,port=4000)