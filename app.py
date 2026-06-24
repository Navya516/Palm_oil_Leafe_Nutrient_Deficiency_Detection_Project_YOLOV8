import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import gdown
import base64


# ---------------- MODEL DOWNLOAD ----------------

MODEL_PATH = "best.pt"

FILE_ID = "1G1nZ-6TPcdAHyIOyo6Mgvrsqsckx37i1"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )


# Load YOLO model
model = YOLO(MODEL_PATH)

def add_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
            linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)
            ),
            url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("bg.jpg")

# ---------------- APP TITLE ----------------

st.title("🌴 Palm Oil Leaf Nutrient Stress Detection")

st.write(
    "Upload a palm oil leaf image to detect nutrient stress"
)


# ---------------- LANGUAGE ----------------

language = st.selectbox(
    "Select Language / భాష ఎంచుకోండి",
    [
        "English",
        "Telugu",
        "Hindi",
        "Tamil"
    ]
)


# ---------------- REMEDIES ----------------

remedies = {

    "boron": {

        "English":
        "Boron deficiency detected. Use Borax or Solubor fertilizer in recommended quantity.",

        "Telugu":
        "బోరాన్ లోపం కనిపించింది. బోరాక్స్ లేదా సోలుబోర్ ఎరువును వాడండి.",

        "Hindi":
        "बोरॉन की कमी पाई गई है। Borax या Solubor खाद उपयोग करें।",

        "Tamil":
        "போரான் குறைபாடு கண்டறியப்பட்டது. Borax அல்லது Solubor பயன்படுத்தவும்."
    },


    "nitrogen": {

        "English":
        "Nitrogen deficiency detected. Use Urea or nitrogen-rich fertilizer.",

        "Telugu":
        "నైట్రోజన్ లోపం కనిపించింది. యూరియా వంటి ఎరువును వాడండి.",

        "Hindi":
        "नाइट्रोजन की कमी है। Urea fertilizer उपयोग करें।",

        "Tamil":
        "நைட்ரஜன் குறைபாடு உள்ளது. Urea பயன்படுத்தவும்."
    },


    "magnesium": {

        "English":
        "Magnesium deficiency detected. Use Magnesium sulphate.",

        "Telugu":
        "మెగ్నీషియం లోపం కనిపించింది. మెగ్నీషియం సల్ఫేట్ వాడండి.",

        "Hindi":
        "मैग्नीशियम की कमी है। Magnesium sulphate उपयोग करें।",

        "Tamil":
        "மக்னீசியம் குறைபாடு உள்ளது. Magnesium sulphate பயன்படுத்தவும்."
    },


    "kalium": {

        "English":
        "Potassium deficiency detected. Use MOP or Potash fertilizer.",

        "Telugu":
        "పొటాషియం లోపం కనిపించింది. పొటాష్ ఎరువును వాడండి.",

        "Hindi":
        "पोटैशियम की कमी है। Potash fertilizer उपयोग करें।",

        "Tamil":
        "பொட்டாசியம் குறைபாடு உள்ளது. Potash உரம் பயன்படுத்தவும்."
    },


    "healthy": {

        "English":
        "Leaf looks healthy. No treatment required.",

        "Telugu":
        "ఆకు ఆరోగ్యంగా ఉంది. మందు అవసరం లేదు.",

        "Hindi":
        "पत्ता स्वस्थ है। किसी उपचार की आवश्यकता नहीं है।",

        "Tamil":
        "இலை ஆரோக்கியமாக உள்ளது. மருந்து தேவையில்லை."
    }
}



# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Choose Palm Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file is not None:


    # Open image
    image = Image.open(uploaded_file)


    # Prediction
    results = model(image)


    # Bounding box image
    detected_image = results[0].plot()



    # SIDE BY SIDE VIEW

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Before 🌿")

        st.image(
            image,
            caption="Original Leaf",
            use_container_width=True
        )



    with col2:

        st.subheader("After 🤖")

        st.image(
            detected_image,
            caption="AI Detection Result",
            use_container_width=True
        )




    # ---------------- RESULT ----------------


    if len(results[0].boxes) > 0:


        for box in results[0].boxes:


            class_id = int(box.cls[0])

            confidence = float(box.conf[0])


            class_name = model.names[class_id]

            class_key = class_name.lower().strip()



            st.success(
                f"Detected Disease: {class_name}"
            )


            st.info(
                f"Confidence: {confidence*100:.2f}%"
            )


            st.subheader(
                "Recommended Solution 💊"
            )


            solution = remedies.get(
                class_key,
                {}
            ).get(
                language,
                "Solution not available"
            )


            st.write(solution)



    else:

        st.warning(
            "No nutrient stress detected"
        )