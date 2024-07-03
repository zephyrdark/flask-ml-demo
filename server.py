import pandas as pd
from flask import Flask, request, jsonify
from waitress import serve
import pickle
# install flask, waitress into your ananconda environment
# use the commands 
#     pip install Flask
#     pip install Waitress

app = Flask(__name__)
modelOne = pickle.load(open('model1.pkl', 'rb')) # load model1 to the server, 'rb' - read binary
df_time_series = pd.read_pickle('model2.pkl') 

@app.route('/model1', methods=['GET'])
def callModelOne():
    xValue = request.args.get('x', type= int)
    return str(modelOne.predict([[xValue]])[0])

@app.route('/model2', methods=['GET'])
def callModelTwo():
   xValue = request.args.get('x', type= int)
   print(df_time_series[xValue])
   return str(df_time_series[xValue])

# run the server
if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8080)
