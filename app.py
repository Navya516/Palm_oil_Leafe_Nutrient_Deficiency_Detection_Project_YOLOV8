import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import base64
import os

MODEL_PATH = "weights/best.pt"
BG_PATH = "sample_images/bg.jpg"

model = YOLO(MODEL_PATH)

def add_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .main-box {{
                background-color: rgba(255,255,255,0.88);
                padding: 25px;
                border-radius: 15px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

add_bg(BG_PATH)

st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.title("🌴 Palm Oil Leaf Disease Detection")

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Telugu"]
)

solutions = {
    "English": {
        "Boron": "Boron deficiency detected. Apply boron-rich fertilizer carefully as per agricultural recommendation.",
        "Healthy": "The leaf looks healthy. Continue regular monitoring and good crop management.",
        "Kalium": "Potassium deficiency detected. Apply potassium-based fertilizer to improve plant strength.",
        "Mg": "Magnesium deficiency detected. Use magnesium sulphate or suitable Mg fertilizer.",
        "Nitrogen": "Nitrogen deficiency detected. Apply nitrogen-rich fertilizer in proper quantity."
    },
    "Hindi": {
        "Boron": "बोरॉन की कमी पाई गई है। कृषि सलाह के अनुसार बोरॉन युक्त उर्वरक का उपयोग करें।",
        "Healthy": "पत्ता स्वस्थ दिखाई दे रहा है। नियमित निगरानी जारी रखें।",
        "Kalium": "पोटैशियम की कमी पाई गई है। पोटैशियम आधारित उर्वरक का उपयोग करें।",
        "Mg": "मैग्नीशियम की कमी पाई गई है। मैग्नीशियम सल्फेट या उपयुक्त Mg उर्वरक का उपयोग करें।",
        "Nitrogen": "नाइट्रोजन की कमी पाई गई है। सही मात्रा में नाइट्रोजन युक्त उर्वरक का उपयोग करें।"
    },
    "Telugu": {
        "Boron": "బోరాన్ లోపం గుర్తించబడింది. వ్యవసాయ సూచనల ప్రకారం బోరాన్ ఎరువును ఉపయోగించండి.",
        "Healthy": "ఆకు ఆరోగ్యంగా ఉంది. పంటను క్రమం తప్పకుండా పరిశీలించండి.",
        "Kalium": "పొటాషియం లోపం గుర్తించబడింది. పొటాషియం ఎరువును ఉపయోగించండి.",
        "Mg": "మెగ్నీషియం లోపం గుర్తించబడింది. మెగ్నీషియం సల్ఫేట్ లేదా సరైన Mg ఎరువును ఉపయోగించండి.",
        "Nitrogen": "నైట్రోజన్ లోపం గుర్తించబడింది. సరైన పరిమాణంలో నైట్రోజన్ ఎరువును ఉపయోగించండి."
    }
}

uploaded_file = st.file_uploader(
    "Upload Palm Oil Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model.predict(source=tmp.name, conf=0.25)

    result = results[0]
    st.image(result.plot(), caption="Detection Result", use_container_width=True)

    if len(result.boxes) > 0:
        st.subheader("Prediction and Solution")

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            st.success(f"Disease: {class_name}")
            st.info(f"Confidence: {confidence:.2f}")
            st.write(solutions[language].get(class_name, "Solution not available."))
    else:
        st.warning("No disease detected.")

st.markdown("</div>", unsafe_allow_html=True)
