#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

X = np.random.rand(100, 1)
y = 2 * X + np.random.randn(100, 1)


from sklearn.linear_model import LinearRegression
import joblib


model = LinearRegression()
model.fit(X, y)


joblib.dump(model, 'model.joblib')


# In[ ]:


from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json['data']


    prediction = model.predict(data)

    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()


# In[ ]:


from reportlab.pdfgen import canvas


def create_pdf():
    c = canvas.Canvas("deployment_steps.pdf")

    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Deployment Steps")
    c.drawString(50, 700, "1. Generated Toy Data")
    c.drawString(50, 680, "2. Saved the Model")
    c.drawString(50, 660, "3. Deployed the Model on Flask")
    c.drawString(50, 640, "4. Created PDF Document")

    c.save()

create_pdf()

