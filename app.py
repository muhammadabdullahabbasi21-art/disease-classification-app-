
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('sample.csv.csv')
print(df.head())
# Separate features (X) and target (y)
X = df.drop(columns=['Disease'])
y = df['Disease']
feature_names = X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print('Training Logistic Regression model...')
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)


train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


def predict_disease(*feature_values):
  
  input_array = np.array([feature_values])
  input_scaled = scaler.transform(input_array)

 
  predicted_class = model.predict(input_scaled)[0]
  probabilities = model.predict_proba(input_scaled)[0]

  
  confidences = {
      str(cls): float(prob)
      for cls, prob in zip(model.classes_, probabilities)
  }
  return predicted_class, confidences



print('Launching Gradio interface...')


input_components = [
    gr.Number(label=col, value=float(df[col].mean())) for col in feature_names
]

demo = gr.Interface(
    fn=predict_disease,
    inputs=input_components,
    outputs=[
        gr.Textbox(label='Predicted Disease'),
        gr.Label(label='Confidence Probabilities'),
    ],
    title='Disease Classification with Logistic Regression',
    description=(
        'Adjust the clinical and biochemical parameters below to predict the'
        ' likely disease category.'
    ),
)

if __name__ == '__main__':
  demo.launch()