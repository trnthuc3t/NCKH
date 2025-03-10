from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Thay đổi chuỗi bí mật cho phù hợp

# Load mô hình SVM và feature extraction
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
feature_extraction_svm = pickle.load(open('feature_extraction_svm.pkl', 'rb'))

# Danh sách người dùng mặc định
users = {
    "admin@gmail.com": "123456",
    "admin2@gmail.com": "123456"
}

# Danh sách lưu trữ các email (tạm thời trong bộ nhớ)
emails = []

def is_spam(content):
    """
    Hàm kiểm tra nội dung thư có phải spam hay không sử dụng mô hình SVM.
    Giả sử: kết quả dự đoán 0 là spam, 1 là ham.
    """
    features = feature_extraction_svm.transform([content])
    prediction = svm_model.predict(features)
    return prediction[0] == 0

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Trang đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and users[email] == password:
            session['user'] = email
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Sai thông tin đăng nhập, vui lòng kiểm tra lại.', 'danger')
    return render_template('login.html')

# Trang đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users:
            flash('Email đã được đăng ký.', 'warning')
        else:
            users[email] = password
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

# Trang Dashboard sau khi đăng nhập
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Vui lòng đăng nhập trước.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

# Trang soạn thư với tích hợp lọc spam bằng SVM
@app.route('/compose', methods=['GET', 'POST'])
def compose():
    if 'user' not in session:
        flash('Vui lòng đăng nhập trước.', 'warning')
        return redirect(url_for('login'))
    if request.method == 'POST':
        recipient = request.form.get('recipient')
        content = request.form.get('content')
        spam_flag = is_spam(content)
        email_data = {
            'sender': session['user'],
            'recipient': recipient,
            'content': content,
            'is_spam': spam_flag
        }
        emails.append(email_data)
        if spam_flag:
            flash('Email đã bị phát hiện là spam và chuyển vào Thư rác.', 'warning')
        else:
            flash('Thư đã được gửi thành công!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('compose.html')

# Trang Hòm thư: hiển thị các email mà người nhận khớp với tài khoản đăng nhập và không bị đánh dấu spam
@app.route('/inbox')
def inbox():
    if 'user' not in session:
        flash('Vui lòng đăng nhập trước.', 'warning')
        return redirect(url_for('login'))
    user_inbox = [email for email in emails if email['recipient'] == session['user'] and not email.get('is_spam', False)]
    return render_template('inbox.html', emails=user_inbox)

# Trang Thư rác: hiển thị các email bị đánh dấu spam
@app.route('/trash')
def trash():
    if 'user' not in session:
        flash('Vui lòng đăng nhập trước.', 'warning')
        return redirect(url_for('login'))
    trash_emails = [email for email in emails if email['recipient'] == session['user'] and email.get('is_spam', False)]
    return render_template('trash.html', emails=trash_emails)

# Đăng xuất
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Đã đăng xuất.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
