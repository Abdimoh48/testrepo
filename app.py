from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("C:/Users/USER1/Pictures/model.h5")


# Define the endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded file from the request
    file = request.files["image"]

    # Open the image file using PIL
    image = Image.open(file)

    # Preprocess the image (resize, normalize, etc.)
    # ... Add your preprocessing code here ...

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Perform the prediction
    prediction = model.predict(np.expand_dims(image_array, axis=0))

    # Process the prediction result
    # ... Add your post-processing code here ...

    # Return the prediction result as a JSON response
    return {"prediction": prediction}


if __name__ == "__main__":
    app.run()
