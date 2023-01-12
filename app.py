from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
	on_thyroxine = float(request.form[' on_thyroxine'])
	query_on_thyroxine'= float(request.form['query_on_thyroxine'])
	on_antithyroid_medication = float(request.form['on_antithyroid_medication'])
	sick = float(request.form['sick'])
	pregnant = float(request.form['pregnant'])
	thyroid_surgery=float(request.form['thyroid_surgery'])
	I131_treatment = float(request.form['I131_treatment'])
	query_hypothyroid=float(request.form['query_hypothyroid'])
	query_hyperthyroid = float(request.form['query_hyperthyroid'])
	lithium = request.form.get('lithium')
	goitre =  request.form.get('goitre')
	tumor =  request.form.get('tumor')
	hypopituitary= request.form.get('hypopituitary')
	psych = float(request.form['psych'])
	T3=float(request.form['T3'])  
	TT4=float(request.form['TT4'])  
	T4U= float(request.form['T4U'])   
	FTI = float(request.form['FTI'])
	referral_source_STMW = float(request.form['referral_source_STMW'])  
	referral_source_SVHC = float(request.form['referral_source_SVHC'])  
	referral_source_SVHD = float(request.form['referral_source_SVHD']) 
	referral_source_SVI = float(request.form['referral_source_SVI'])   
	referral_source_other = float(request.form['referral_source_other'])  

        
       data = np.array([[age, sex,on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,query_hyperthyroid,lithium,
       goitre,tumor,hypopituitary, psych, T3,TT4,T4U,FTI,referral_source_STMW,referral_source_SVHC,referral_source_SVHD,referral_source_SVI,referral_source_other]])
        
       my_prediction = model.predict(data)
        
       return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)
