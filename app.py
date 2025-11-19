import streamlit as st
from PIL import Image
import io
import pandas as pd
import face_utils
import glob

st.title("AI Attendance System (DeepFace Version)")

menu = st.sidebar.selectbox("Menu", ["Register", "Attendance", "Students", "Records"])


# REGISTER STUDENT
if menu == "Register":
    name = st.text_input("Name")
    roll = st.text_input("Roll")
    photo = st.file_uploader("Upload Photo", type=["jpg", "png"])

    if st.button("Register"):
        if not (name and roll and photo):
            st.warning("Fill all fields")
        else:
            img = Image.open(photo)
            ok, msg = face_utils.register_student(name, roll, img)
            st.success(msg) if ok else st.error(msg)


# ATTENDANCE
if menu == "Attendance":
    img_file = st.camera_input("Capture Photo")

    if img_file:
        pil = Image.open(io.BytesIO(img_file.getvalue()))
        emb = face_utils.get_embedding(pil)

        if emb is None:
            st.error("No face detected")
        else:
            roll, sim = face_utils.match_face(emb)
            if roll:
                data = face_utils.load_data()
                name = data[roll]["name"]
                face_utils.mark_attendance(roll, name)
                st.success(f"Present: {name} ({roll})")
            else:
                st.warning("Face not recognized")


# STUDENTS
if menu == "Students":
    data = face_utils.load_data()
    if not data:
        st.info("No students")
    else:
        rows = [{"roll": r, "name": v["name"], "photo": v["photo"]} for r, v in data.items()]
        st.dataframe(pd.DataFrame(rows))


# ATTENDANCE RECORDS
if menu == "Records":
    files = glob.glob("attendance_*.csv")
    if not files:
        st.info("No records")
    else:
        file = st.selectbox("Select file", files)
        st.dataframe(pd.read_csv(file))
