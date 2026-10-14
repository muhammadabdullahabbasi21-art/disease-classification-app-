
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv")

# Select input and output columns
X = df["Message"]
y = df["Category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Prediction function
def predict_spam(message):

    if not message.strip():
        return "Please enter a message."

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]

    if prediction == "spam":
        return "Spam Message"
    else:
        return "Ham (Not Spam)"


# Create Gradio interface
app = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter your SMS message here..."
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Spam Message Classifier",
    description="Logistic Regression Model using Scikit-learn"
)

# Launch application
app.launch(share=True)
