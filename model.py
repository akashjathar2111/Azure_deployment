import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("Toyoto_Corrola.csv")
data = df.iloc[:,2:]
data = data.rename({'Age_08_04':'Age'},axis = 1)
data = data.drop(df.index[[110,110,221,960,991]],axis = 0).reset_index()
X = data[['Age','KM','HP','Cylinders','Gears','Weight']]
Y = data['Price']
reg = LinearRegression()
reg.fit(X,Y)
pickle.dump(reg,open('model.pkl','wb'))

#model = pickle.load(open('model.pkl','rb'))

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template("index1.html", prediction_text='Your Car price will be {}'.format(output))

@app.route('/predict_api',methods=['POST'])

def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
