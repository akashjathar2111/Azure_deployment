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

model = pickle.load(open('model.pkl','rb'))


