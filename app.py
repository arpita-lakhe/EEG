import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------
# App Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Eye State Detection",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Eye State Detection App")
st.markdown(
    "Upload an **eye image** and the model will predict whether the eye is **Open** or **Closed**."
)

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# -------------------------------------------------
# Image Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------------------------
    # Preprocess Image
    # -------------------------------------------------
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    st.subheader("Prediction Result")

    if confidence > 0.5:
        st.success(f"👀 Eye State: **OPEN**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.error(f"😴 Eye State: **CLOSED**")
        st.write(f"Confidence: **{1 - confidence:.2f}**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "📌 **Project:** Eye State Detection using CNN & Streamlit  \n"
    "🎓 **Domain:** Machine Learning / Deep Learning"
)
