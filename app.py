import numpy as np
from flask import Flask, request, render_template
import pickle
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    

    input = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    prediction = model.predict(final_input)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', output='Paitent has thyroid $ :{}'.format(output))
   

if __name__ == "__main__":
    app.run(debug=True)
