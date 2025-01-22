from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forum.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the model and tokenizer
model_path = r'C:\Users\Personal Computer\Desktop\New folder\Bert_model'
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=4)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Define database models
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    post_id = db.Column(db.Integer, nullable=False)
    original_comment = db.Column(db.Text, nullable=False)
    filtered_comment = db.Column(db.Text, nullable=False)

# Preprocessing and profanity detection
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_and_filter_profanity(comment):
    severity_levels = ['High', 'Mild', 'Moderate', 'Nonprofane']
    filtered_comment = comment

    # Preprocess text
    preprocessed_comment = preprocess_text(comment)

    # Tokenize and predict
    inputs = tokenizer.encode_plus(
        preprocessed_comment,
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=32
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    # Check if comment contains profanity
    if severity_levels[prediction] != 'Nonprofane':
        profane_words = ['gago', 'putangina', 'tanginamo', 'tarantado']  # Add your list of profane words
        for word in profane_words:
            filtered_comment = filtered_comment.replace(word, word[0] + '*' * (len(word) - 1))
    
    return filtered_comment, severity_levels[prediction]

@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to add a comment
@app.route('/add_comment', methods=['POST'])
def add_comment():
    data = request.json
    user_id = data.get('user_id')
    post_id = data.get('post_id')
    comment = data.get('comment')

    if not user_id or not post_id or not comment:
        return jsonify({'error': 'Invalid input'}), 400

    # Detect and filter profanity
    filtered_comment, severity = detect_and_filter_profanity(comment)

    # Save to database
    new_comment = Comment(
        user_id=user_id,
        post_id=post_id,
        original_comment=comment,
        filtered_comment=filtered_comment
    )
    db.session.add(new_comment)
    db.session.commit()

    return jsonify({
        'message': 'Comment added successfully',
        'filtered_comment': filtered_comment,
        'severity': severity
    })

# API endpoint to fetch comments
@app.route('/get_comments/<int:post_id>', methods=['GET'])
def get_comments(post_id):
    comments = Comment.query.filter_by(post_id=post_id).all()
    result = [
        {
            'user_id': comment.user_id,
            'original_comment': comment.original_comment,
            'filtered_comment': comment.filtered_comment
        }
        for comment in comments
    ]
    return jsonify(result)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

