from flask import Flask, render_template, request, redirect, url_for, session
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pickle
import os
import time

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
app = Flask(__name__)
app.secret_key = 'super-secret-key-12345'

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'

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

def get_gmail_service():
    creds = None
    try:
        if 'credentials' in session:
            creds = Credentials(**session['credentials'])
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            session['credentials'] = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }
        return build('gmail', 'v1', credentials=creds)
    except OSError as e:
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write(f"OSError in get_gmail_service: {str(e)}\n")
        raise
    except Exception as e:
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write(f"Error in get_gmail_service: {str(e)}\n")
        raise

def predict_mail(model_name, input_text):
    input_user_mail = [input_text]
    feature_extraction = pickle.load(open(feature_extraction_files[model_name], 'rb'))
    input_data_features = feature_extraction.transform(input_user_mail)
    model = models.get(model_name, models['svm'])
    prediction = model.predict(input_data_features)
    return prediction[0]

def get_latest_email(service):
    for attempt in range(3):
        try:
            results = service.users().messages().list(userId='me', maxResults=1).execute()
            messages = results.get('messages', [])
            if messages:
                msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
                return msg['snippet']
            return None
        except (OSError, HttpError) as e:
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(f"Error in get_latest_email (attempt {attempt + 1}): {str(e)}\n")
            if attempt < 2:
                time.sleep(2)
            else:
                raise

@app.route('/')
def index():
    if 'credentials' not in session:
        return redirect(url_for('login'))
    service = get_gmail_service()
    latest_email = get_latest_email(service)
    return render_template('index.html', latest_email=latest_email, models=models.keys())

@app.route('/login')
def login():
    return redirect(url_for('authorize'))

@app.route('/authorize')
def authorize():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    flow.redirect_uri = url_for('oauth2callback', _external=True)
    authorization_url, state = flow.authorization_url(access_type='offline')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    try:
        state = session.get('state')
        if not state:
            return "Error: State not found in session", 400
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES, state=state)
        flow.redirect_uri = url_for('oauth2callback', _external=True)
        authorization_response = request.url
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write(f"Authorization response: {authorization_response}\n")
        flow.fetch_token(authorization_response=authorization_response)
        creds = flow.credentials
        session['credentials'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write("Credentials saved to session\n")
        return redirect(url_for('index'))
    except Exception as e:
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write(f"Error in oauth2callback: {str(e)}\n")
        return f"Error during authentication: {str(e)}", 500

@app.route('/analyze', methods=['POST'])
def analyze_mail():
    service = get_gmail_service()
    latest_email = get_latest_email(service)
    if request.method == 'POST':
        model_name = request.form.get('model')
        classify = predict_mail(model_name, latest_email)
        return render_template('index.html', latest_email=latest_email, classify=classify, models=models.keys())
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='localhost')