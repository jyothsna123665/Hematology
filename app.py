import os
import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from pdfkit import pdfkit
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fpdf import FPDF
import pdfkit

app = Flask(__name__)
app.secret_key='Hematology'
# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your_secret_key'  # For sessions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model
model = load_model('C:/Users/gowri/PycharmProjects/Hematology/split_data/vgg_blood_model.h5')
img_size = (224, 224)

# Blood group image mapping
blood_group_image_map = {
    'A+': 'static/blood_groups/A+.png',
    'B+': 'static/blood_groups/B+.png',
    'AB+': 'static/blood_groups/AB+.png',
    'O+': 'static/blood_groups/O+.png',
    'A-': 'static/blood_groups/A-.png',
    'B-': 'static/blood_groups/B-.png',
    'AB-': 'static/blood_groups/AB-.png',
    'O-': 'static/blood_groups/O-.png'
}


# --- Prediction Function ---
def predict_blood_group(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    labels = {0: 'A+', 1: 'B+', 2: 'AB+', 3: 'O+', 4: 'A-', 5: 'B-', 6: 'AB-', 7: 'O-'}
    return labels[np.argmax(prediction)]


# --- PDF Generation ---
def generate_pdf(user_data, predicted_class, report_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Blood Group Prediction Report", ln=True, align='C')
    pdf.ln(10)

    now = datetime.datetime.now()
    pdf.cell(200, 10, txt=f"Date: {now.strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Time: {now.strftime('%H:%M:%S')}", ln=True)
    pdf.ln(10)

    if user_data:
        pdf.cell(200, 10, txt=f"Name: {user_data.get('name')}", ln=True)
        pdf.cell(200, 10, txt=f"Email: {user_data.get('email')}", ln=True)
        pdf.cell(200, 10, txt=f"Phone: {user_data.get('phone')}", ln=True)
        pdf.cell(200, 10, txt=f"Address: {user_data.get('address')}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {user_data.get('age')}", ln=True)
        pdf.ln(10)

    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt=f"Predicted Blood Group: {predicted_class}", ln=True)
    pdf.output(report_path)


# --- Routes ---

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/account')
def account():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    session['user_data'] = user_data
    return render_template('account.html', user_data=session['user_data'])


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('welcome'))


@app.route('/procedure')
def procedure():
    return render_template('procedure.html')


@app.route('/details_known')
def details_known():
    return redirect(url_for('login'))


@app.route('/details_unknown')
def details_unknown():
    # User directly uploads without providing details
    return render_template('upload.html', user_data=None)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user_data'] = {
            'name': request.form['name'],
            'email': request.form['email'],
            'phone': request.form['phone'],
            'address': request.form['address'],
            'age': request.form['age']
        }
        return redirect(url_for('upload'))
    return render_template('login.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    user_data = session.get('user_data', None)

    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            # Predict blood group
            predicted_class = predict_blood_group(filepath)
            blood_image = blood_group_image_map.get(predicted_class)

            # Get compatibility info
            blood_compatibility = {
                'O-': {'giving_blood_group': 'Everyone', 'receiving_blood_group': 'O-'},
                'O+': {'giving_blood_group': 'O+, A+, B+, AB+', 'receiving_blood_group': 'O-, O+'},
                'A-': {'giving_blood_group': 'A-, A+, AB-, AB+', 'receiving_blood_group': 'A-, O-'},
                'A+': {'giving_blood_group': 'A+, AB+', 'receiving_blood_group': 'A-, A+, O-, O+'},
                'B-': {'giving_blood_group': 'B-, B+, AB-, AB+', 'receiving_blood_group': 'B-, O-'},
                'B+': {'giving_blood_group': 'B+, AB+', 'receiving_blood_group': 'B-, B+, O-, O+'},
                'AB-': {'giving_blood_group': 'AB-, AB+', 'receiving_blood_group': 'AB-, A-, B-, O-'},
                'AB+': {'giving_blood_group': 'AB+', 'receiving_blood_group': 'Everyone'}
            }

            giving_blood_group = blood_compatibility.get(predicted_class, {}).get('giving_blood_group', 'Unknown')
            receiving_blood_group = blood_compatibility.get(predicted_class, {}).get('receiving_blood_group', 'Unknown')

            # Generate PDF
            now = datetime.datetime.now()
            report_filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join('static/reports', report_filename)
            generate_pdf(user_data, predicted_class, report_path)

            return render_template(
                'bloodreport.html',
                blood_group=predicted_class,
                blood_image=blood_image,
                report_path=report_path,
                user_data=user_data,
                now=now,
                giving_blood_group=giving_blood_group,
                receiving_blood_group=receiving_blood_group
            )

    return render_template('upload.html', user_data=user_data, now=datetime.datetime.now())


@app.route('/report')
def report():
    blood_compatibility = {
        'O-': {'giving_blood_group': 'Everyone', 'receiving_blood_group': 'O-'},
        'O+': {'giving_blood_group': 'O+, A+, B+, AB+', 'receiving_blood_group': 'O-, O+'},
        'A-': {'giving_blood_group': 'A-, A+, AB-, AB+', 'receiving_blood_group': 'A-, O-'},
        'A+': {'giving_blood_group': 'A+, AB+', 'receiving_blood_group': 'A-, A+, O-, O+'},
        'B-': {'giving_blood_group': 'B-, B+, AB-, AB+', 'receiving_blood_group': 'B-, O-'},
        'B+': {'giving_blood_group': 'B+, AB+', 'receiving_blood_group': 'B-, B+, O-, O+'},
        'AB-': {'giving_blood_group': 'AB-, AB+', 'receiving_blood_group': 'AB-, A-, B-, O-'},
        'AB+': {'giving_blood_group': 'AB+', 'receiving_blood_group': 'Everyone'}
    }

    # Get compatible blood groups based on the detected blood group
    giving_blood_group = blood_compatibility.get(blood_group, {}).get('giving_blood_group', 'Unknown')
    receiving_blood_group = blood_compatibility.get(blood_group, {}).get('receiving_blood_group', 'Unknown')

    return render_template('report.html',
                           now=datetime.now(),
                           user_data=user_data,
                           blood_group=blood_group,
                           blood_compatibility=blood_compatibility,
                           giving_blood_group=giving_blood_group,
                           receiving_blood_group=receiving_blood_group)



# --- Run App ---
if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/reports', exist_ok=True)
    app.run(debug=True)
