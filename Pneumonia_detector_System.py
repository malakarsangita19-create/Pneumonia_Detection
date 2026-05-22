import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from fpdf import FPDF
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp{
background: linear-gradient(to right, #dbeafe, #eff6ff);
}
.title{
font-size:40px;
font-weight:bold;
color:#1e3a8a;
}
</style>
""",unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history=[]

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if "role" not in st.session_state:
    st.session_state.role=""

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:

    st.markdown("<div class='title'>AI Pneumonia Detection System</div>",unsafe_allow_html=True)

    role=st.selectbox("Select Role",["Doctor","User"])
    username=st.text_input("Username")
    password=st.text_input("Password",type="password")

    if st.button("Login"):

        if role=="Doctor" and username=="doctor" and password=="1234":
            st.session_state.logged_in=True
            st.session_state.role="Doctor"
            st.rerun()

        elif role=="User" and username=="user" and password=="0987":
            st.session_state.logged_in=True
            st.session_state.role="User"
            st.rerun()

        else:
            st.error("Invalid Login")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🫁 Pneumonia Detector")
st.sidebar.success(f"Logged in as: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in=False
    st.rerun()

# ---------------- HEADER ----------------
st.markdown("<div class='title'>AI Pneumonia Detection Dashboard</div>",unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")

model=load_model()

def preprocess(img):
    img=img.convert("RGB").resize((150,150))
    arr=np.array(img)/255
    return np.expand_dims(arr,0)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Info",
    "X-ray Analysis",
    "Doctor Finder",
    "Medical Report",
    "Doctor Dashboard"
])

# ---------------- TAB 1 ----------------
with tab1:
    if st.session_state.role == "Doctor":
        st.warning("Doctors cannot enter patient data.")
    else:
        name=st.text_input("Patient Name")
        age=st.number_input("Age",1,120)
        gender=st.selectbox("Gender",["Male","Female","Other"])

        height=st.number_input("Height (meters)",1.0,2.5,1.6)
        weight=st.number_input("Weight (kg)",30,150,60)

        bmi=weight/(height**2)
        st.info(f"BMI Score : {round(bmi,2)}")

        st.session_state.name=name
        st.session_state.age=age
        st.session_state.gender=gender
        st.session_state.bmi=round(bmi,2)

        fever=st.checkbox("Fever")
        cough=st.checkbox("Cough")
        breath=st.checkbox("Breathing Difficulty")
        chest=st.checkbox("Chest Pain")

# ---------------- TAB 2 ----------------
with tab2:
    if st.session_state.role == "Doctor":
        st.warning("Doctors cannot upload X-rays.")
    else:
        file=st.file_uploader("Upload Image",["jpg","png","jpeg"])

        if file:
            img=Image.open(file)
            st.image(img,width=300)

            st.session_state.last_image = img

            with st.spinner("Analyzing..."):
                time.sleep(2)

            data=preprocess(img)
            pred=model.predict(data)
            prob=float(pred[0][0])

            result="Pneumonia" if prob>0.5 else "Normal"

            # -------- SEVERITY LOGIC --------
            if prob < 0.4:
                severity = "Low"
                color = "green"
            elif prob < 0.7:
                severity = "Moderate"
                color = "orange"
            else:
                severity = "High"
                color = "red"

            st.session_state.last_result = result
            st.session_state.severity = severity
            st.session_state.severity_color = color
            st.session_state.lung_score = max(0, min(100, int((1 - prob) * 100)))

            st.subheader("Result")
            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {round(prob*100,2)}%")

            st.markdown(
                f"### Severity Level: <span style='color:{color}; font-weight:bold'>{severity}</span>",
                unsafe_allow_html=True
            )

            # -------- DOs & DON'Ts --------
            st.subheader("Guidelines")

            if result == "Pneumonia":
                st.warning("""
DOs:
- Take prescribed medicines
- Drink warm fluids
- Take proper rest

DON'Ts:
- Avoid smoking
- Avoid cold drinks
- Avoid dust exposure
                """)
            else:
                st.success("No Pneumonia Detected. You are safe.")

            # Save history
            st.session_state.history.append({
                "Patient":st.session_state.get("name","Unknown"),
                "Age":st.session_state.get("age",""),
                "Gender":st.session_state.get("gender",""),
                "BMI":st.session_state.get("bmi",""),
                "Result":result
            })

# ---------------- TAB 3 ----------------
with tab3:
    if st.session_state.role == "Doctor":
        st.warning("Doctors cannot find.")
    else:
        st.subheader("Find a Doctor")

        doctors = pd.DataFrame({
            "Name": ["Dr. Sharma", "Dr. Roy", "Dr. Khan", "Dr. Das"],
            "Specialization": ["Pulmonologist", "General Physician", "Pulmonologist", "General Physician"],
            "Location": ["Kolkata", "Delhi", "Mumbai", "Kolkata"],
            "Contact": ["9876543210", "9876500000", "9123456780", "9988776655"]
        })

        spec = st.selectbox("Specialization", doctors["Specialization"].unique())
        city = st.selectbox("City", doctors["Location"].unique())

        filtered = doctors[
            (doctors["Specialization"] == spec) &
            (doctors["Location"] == city)
        ]

        if not filtered.empty:
            st.dataframe(filtered)
        else:
            st.warning("No doctors found")

# ---------------- TAB 4 ----------------
with tab4:
    if st.session_state.role == "Doctor":
        st.warning("Doctors cannot generate reports.")
    else:
        st.subheader("Generate Medical Report")

        if st.button("Generate PDF Report"):

            prediction = st.session_state.get("last_result", "Not analyzed")
            severity = st.session_state.get("severity", "Unknown")
            lung = st.session_state.get("lung_score", "Unknown")

            name = st.session_state.get("name", "Unknown")
            age = st.session_state.get("age", "Unknown")
            gender = st.session_state.get("gender", "Unknown")
            bmi = st.session_state.get("bmi", "Unknown")

            img = st.session_state.get("last_image", None)

            if img:
                img_path = "xray_image.png"
                img.save(img_path)

            pdf = FPDF()
            pdf.add_page()

            try:
                pdf.image("logo1.jpeg", x=10, y=8, w=30)
            except:
                pass

            pdf.set_font("Arial","B",18)
            pdf.cell(0,10,"AI PNEUMONIA DETECTION REPORT",ln=True,align="C")

            pdf.ln(10)

            pdf.set_font("Arial","",12)
            pdf.cell(0,8,"Hospital: AI Medical Research Center",ln=True)

            pdf.ln(5)

            pdf.set_font("Arial","B",14)
            pdf.cell(0,10,"Patient Information",ln=True)

            pdf.set_font("Arial","",12)
            pdf.cell(50,8,"Patient Name:",1)
            pdf.cell(0,8,str(name),1,1)

            pdf.cell(50,8,"Age:",1)
            pdf.cell(0,8,str(age),1,1)

            pdf.cell(50,8,"Gender:",1)
            pdf.cell(0,8,str(gender),1,1)

            pdf.cell(50,8,"BMI:",1)
            pdf.cell(0,8,str(bmi),1,1)

            pdf.ln(5)

            pdf.set_font("Arial","B",14)
            pdf.cell(0,10,"AI Diagnosis Result",ln=True)

            pdf.set_font("Arial","",12)

            pdf.cell(50,8,"Prediction:",1)
            pdf.cell(0,8,prediction,1,1)

            # -------- COLORED SEVERITY IN PDF --------
            if severity == "Low":
                pdf.set_text_color(0,150,0)
            elif severity == "Moderate":
                pdf.set_text_color(255,165,0)
            else:
                pdf.set_text_color(255,0,0)

            pdf.cell(50,8,"Severity:",1)
            pdf.cell(0,8,severity,1,1)

            pdf.set_text_color(0,0,0)

            pdf.cell(50,8,"Lung Score:",1)
            pdf.cell(0,8,str(lung)+"/100",1,1)

            pdf.ln(5)

            if img:
                pdf.cell(0,10,"X-ray Image:",ln=True)
                pdf.image(img_path,x=55,w=100)

            pdf.ln(10)

            pdf.set_font("Arial","B",14)
            pdf.cell(0,10,"Guidelines",ln=True)

            pdf.set_font("Arial","",12)

            if prediction=="Pneumonia":
                pdf.multi_cell(0,8,
                "DOs:\n- Take medicines\n- Rest\n- Drink fluids\n\nDON'Ts:\n- No smoking\n- Avoid cold items")
            else:
                pdf.multi_cell(0,8,"No Pneumonia Detected. You are safe.")

            pdf.output("pneumonia_report.pdf")

            with open("pneumonia_report.pdf","rb") as f:
                st.download_button("Download Report",f,"pneumonia_report.pdf")

# ---------------- TAB 5 ----------------
with tab5:
    if st.session_state.role != "Doctor":
        st.warning("Only doctors can access this dashboard.")
    else:
        st.subheader("Doctor Dashboard")

        search = st.text_input("Search Patient")

        if st.session_state.history:
            df=pd.DataFrame(st.session_state.history)

            if search:
                df=df[df["Patient"].str.contains(search,case=False)]

            if not df.empty:
                st.dataframe(df)

                col1,col2,col3,col4=st.columns(4)

                total=len(df)
                pne=df[df["Result"]=="Pneumonia"].shape[0]
                normal=df[df["Result"]=="Normal"].shape[0]

                percentage = (pne/total)*100 if total>0 else 0

                col1.metric("Total",total)
                col2.metric("Pneumonia",pne)
                col3.metric("Normal",normal)
                col4.metric("Pneumonia %",f"{round(percentage,2)}%")

            else:
                st.warning("No data found")
        else:
            st.info("No history")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("AI Pneumonia Detection System")