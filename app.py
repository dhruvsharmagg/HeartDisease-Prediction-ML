import pandas as pd
import joblib
import numpy as np
from flask_mysqldb import MySQL
from flask import Flask, request, jsonify, render_template, request
import pickle


app = Flask(__name__)
model = joblib.load('xgboost.p')

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'heart'

mysql = MySQL(app)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/parameters')
def parameters():
    return render_template('parameters.html')

@app.route('/idea')
def idea():
    return render_template('idea.html')

@app.route('/about')
def about():
    return render_template('about.html')    

@app.route('/')
def form():
    return render_template('form.html')


@app.route('/',methods=['POST'])
def data():
    if request.method == "POST":
        male = np.float(request.form['male'])
        age = np.float(request.form['age'])
        #education = np.float(request.form['education'])
        currentSmoker = np.float(request.form['currentSmoker'])
        cigsPerDay = np.float(request.form['cigsPerDay'])
        BPMeds = np.float(request.form['BPMeds'])
        totChol = np.float(request.form['totChol'])
        sysBP = np.float(request.form['sysBP'])
        diaBP = np.float(request.form['diaBP'])
        BMI = np.float(request.form['BMI'])
        heartRate = np.float(request.form['heartRate'])
        glucose = np.float(request.form['glucose'])

       

        df = pd.DataFrame({'male': [male], 'age': [age],  'currentSmoker': [currentSmoker], 'cigsPerDay':[cigsPerDay], 'BPMeds':[BPMeds], 'totChol':[totChol], 'sysBP':[sysBP], 'diaBP':[diaBP], 'BMI':[BMI], 'heartRate':[heartRate], 'glucose':[glucose]})
        #input=[male,age,education,currentSmoker,cigsPerDay,BPMeds,totChol,sysBP,diaBP,BMI,heartRate,glucose]

        #df_ip = pd.DataFrame(input)
        #df_ip.columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate','glucose']
        output = model.predict(df)

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO heartdata(male,age,currentSmoker,cigsPerDay,BPMeds,totChol,sysBP,diaBP,BMI,heartRate,glucose,TenYearCHD) VALUES (%s, %s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s)", (male, age,currentSmoker,cigsPerDay,BPMeds,totChol,sysBP,diaBP,BMI,heartRate,glucose,output))  #insert into tablename(table column name) (datatypes) (all the variables)
        mysql.connection.commit()
        cur.close()
    
    if int(output)== 1: 
        prediction ='Patient will develop Heart Disease'
    else: 
        prediction ='Patient will not develop Heart Disease'

    return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

