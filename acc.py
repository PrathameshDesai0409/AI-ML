from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('keras_model.h5')

# Load your test data
# Replace X_test and y_test with your test features and labels
X_test, y_test = ...

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
