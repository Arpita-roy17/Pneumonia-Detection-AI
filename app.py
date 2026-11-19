import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.image(
    "https://img.icons8.com/color/96/stethoscope.png",
    width=80
)

st.sidebar.title("🏥 AI Pneumonia Detection")

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Model Information")

st.sidebar.success("DenseNet121")

st.sidebar.write("Transfer Learning")

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Dataset")

st.sidebar.write("Chest X-Ray Images (Pneumonia)")
st.sidebar.write("• NORMAL")
st.sidebar.write("• PNEUMONIA")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Dataset Statistics")

st.sidebar.table({
    "Feature":[
        "Training",
        "Validation",
        "Testing",
        "Classes"
    ],
    "Value":[
        "4173",
        "1043",
        "624",
        "2"
    ]
})

st.sidebar.markdown("---")

st.sidebar.info("""
👩‍💻 **Developer**

Arpita

B.Tech CSE (AI & ML)
""")

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

/* Don't force the app background.
   Let Streamlit handle Light/Dark mode automatically */

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:white;
}

.subtitle{
    font-size:18px;
    text-align:center;
    color:#F8F9FA;
}

.hero{
    background:linear-gradient(135deg,#0B5ED7,#4AA3FF);
    padding:30px;
    border-radius:18px;
    margin-bottom:25px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.15);
}

/* Metrics */

div[data-testid="stMetric"]{
    border-radius:15px;
    padding:15px;
    border:1px solid rgba(128,128,128,0.25);
}

/* Predict button */

.stButton > button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:15px;
}

/* File uploader */

div[data-testid="stFileUploader"]{
    border:2px dashed #4AA3FF;
    border-radius:15px;
    padding:15px;
}

/* Tables */

[data-testid="stTable"]{
    border-radius:15px;
}

[data-testid="stDataFrame"]{
    border-radius:15px;
}

/* Hide Streamlit menu */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO
# =====================================

st.markdown("""
<div class="hero">

<div class="main-title">

🏥 AI Clinical Decision Support System

</div>

<br>

<div class="subtitle">

Early Detection of Pneumonia using Deep Learning

<br>

DenseNet121 Transfer Learning Model

</div>

</div>
""", unsafe_allow_html=True)


# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    return tf.keras.layers.TFSMLayer(
        "models/densenet_savedmodel",
        call_endpoint="serve"
    )

model = load_model()

st.success("✅ DenseNet121 Model Loaded Successfully")

# =====================================
# UPLOAD
# =====================================

st.markdown("## 📂 Upload Chest X-Ray")

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"],
    help="Upload Chest X-Ray Image"
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing Chest X-Ray..."):

            img = image.resize((224,224))
            img = np.array(img)
            img = img.astype(np.float32)/255.0
            img = np.expand_dims(img,axis=0)

            prediction = model(img)
            prediction = prediction.numpy()

            probability = prediction[0][0]

            if probability > 0.5:

                predicted_class = "PNEUMONIA"
                confidence = probability*100

            else:

                predicted_class = "NORMAL"
                confidence = (1-probability)*100

        st.markdown("---")

        st.subheader("📋 Prediction Result")

        if predicted_class=="PNEUMONIA":

            st.error("🦠 Prediction : PNEUMONIA")

        else:

            st.success("✅ Prediction : NORMAL")

        st.metric(
    "Prediction Confidence",
    f"{float(confidence):.2f}%"
)

        st.write("### Confidence Meter")

        st.progress(float(confidence)/100)

        st.write(f"**Prediction Confidence : {confidence:.2f}%**")

        progress_value = float(confidence) / 100
        st.progress(progress_value)

        # =====================================
        # AI DIAGNOSIS REPORT
        # =====================================

        st.markdown("---")
        st.subheader("🩺 AI Diagnosis Report")

        # Confidence Level
        if confidence >= 95:
            confidence_level = "🟢 Very High"

        elif confidence >= 85:
            confidence_level = "🟢 High"

        elif confidence >= 70:
            confidence_level = "🟡 Moderate"

        else:
            confidence_level = "🟠 Low"

        if predicted_class == "NORMAL":

            st.success(f"""

### Diagnosis

The uploaded Chest X-ray has been analyzed using the **DenseNet121 Deep Learning Model**.

The AI model predicts that the image belongs to the **NORMAL** category.

### Prediction Confidence

**{confidence:.2f}%**

### Confidence Level

{confidence_level}

### Observation

• No significant radiographic patterns suggestive of pneumonia were identified.

• Lung fields appear clear based on the learned features of the AI model.

• This prediction should always be interpreted together with clinical symptoms and professional medical evaluation.

""")

            st.info("""

### 💡 Medical Recommendation

✔ Continue routine monitoring if symptoms persist.

✔ If fever, cough, chest pain or breathing difficulty develops, consult a physician.

✔ This AI tool is intended to assist diagnosis and **does not replace a radiologist.**

""")

        else:

            st.error(f"""

### Diagnosis

The uploaded Chest X-ray has been analyzed using the **DenseNet121 Deep Learning Model**.

The AI model predicts the presence of **PNEUMONIA**.

### Prediction Confidence

**{confidence:.2f}%**

### Confidence Level

{confidence_level}

### Observation

• The detected radiographic patterns are consistent with pneumonia.

• Further clinical evaluation and radiologist review are recommended.

• This prediction should be confirmed by a qualified medical professional.

""")

            st.warning("""

### ⚠ Medical Recommendation

• Immediate consultation with a physician or radiologist is recommended.

• Additional laboratory tests and clinical examination may be required.

• Do not rely solely on this AI prediction for medical decisions.

""")

        report = f"""
AI PNEUMONIA DETECTION REPORT

Prediction : {predicted_class}

Confidence : {confidence:.2f}%

Model : DenseNet121

Generated by AI Clinical Decision Support System
"""

        st.download_button(
            "📄 Download AI Report",
            report,
            file_name="AI_Report.txt"
        )        


        # =====================================
        # MODEL INFORMATION
        # =====================================

        st.markdown("---")

        st.subheader("🧠 Model Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Architecture", "DenseNet121")
            st.metric("Input Size", "224 × 224")

        with col2:
            st.metric("Framework", "TensorFlow")
            st.metric("Classes", "2")

        # =====================================
        # DATASET INFORMATION
        # =====================================

        st.markdown("---")

        st.subheader("📂 Dataset Information")

        st.dataframe(
    {
        "Feature": [
            "Dataset",
            "Training Images",
            "Validation Images",
            "Testing Images",
            "Image Size",
            "Classes",
            "Model"
        ],
        "Value": [
            "Chest X-Ray Images (Pneumonia)",
            "4,173",
            "1,043",
            "624",
            "224 × 224 RGB",
            "NORMAL / PNEUMONIA",
            "DenseNet121 (Transfer Learning)"
        ]
    },
    hide_index=True,
    use_container_width=True
)

        # =====================================
        # DISCLAIMER
        # =====================================

        st.markdown("---")

        st.info("""
### ⚠ Disclaimer

This AI system is developed for **educational and research purposes only**.

The prediction generated by the model **must not be considered as a final medical diagnosis**.

Clinical decisions should always be made by qualified healthcare professionals after reviewing the patient's medical history, symptoms, laboratory findings and radiological examination.
""")

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
"""
<center>

Developed using ❤️ Streamlit • TensorFlow • DenseNet121

<br>

B.Tech CSE (AI & ML)

</center>
""",
unsafe_allow_html=True
)