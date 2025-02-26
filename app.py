from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load tất cả các mô hình
models = {
    'logistic_regression': pickle.load(open('logistic_regression.pkl', 'rb')),
    'random_forest': pickle.load(open('random_forest_model.pkl', 'rb')),
    'naive_bayes': pickle.load(open('naive_bayes_model.pkl', 'rb')),
    'svm': pickle.load(open('svm_model.pkl', 'rb')),
    'decision_tree': pickle.load(open('decision_tree_model.pkl', 'rb'))
}

feature_extraction_files = {
    'logistic_regression': 'feature_extraction_logistic_regressio.pkl',
    'random_forest': 'feature_extraction_random_forest.pkl',
    'naive_bayes': 'feature_extraction_naive_bayes.pkl',
    'svm': 'feature_extraction_svm.pkl',
    'decision_tree': 'feature_extraction_decision_tree.pkl'
}

def predict_mail(model_name, input_text):
    input_user_mail = [input_text]
    feature_extraction = pickle.load(open(feature_extraction_files[model_name], 'rb'))
    input_data_features = feature_extraction.transform(input_user_mail)
    model = models.get(model_name, models['svm'])
    prediction = model.predict(input_data_features)
    return prediction[0]



@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    classify = None
    selected_model = 'svm'
    if request.method == 'POST':
        mail = request.form.get('mail')
        model_name = request.form.get('model')
        classify = predict_mail(model_name, mail)
        selected_model = model_name

    return render_template('index.html', classify=classify, selected_model=selected_model, models=models.keys())


if __name__ == '__main__':
    app.run(debug=True)
