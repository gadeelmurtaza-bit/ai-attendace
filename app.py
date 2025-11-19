import streamlit as st
from PIL import Image
import io
import pandas as pd
import face_utils
import glob

st.title("AI Attendance System (OpenCV Version)")

menu = st.sidebar.selectbox("Menu", ["Register", "Bulk Register", "Attendance", "Students", "Records"])


# Register
if menu == "Register":
    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    photo = st.file_uploader("Upload Photo", type=["jpg", "png"])

    if st.button("Register"):
        if photo:
            img = Image.open(photo)
            ok, msg = face_utils.register_student(name, roll, img)
            st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Upload a photo")


# Bulk
if menu == "Bulk Register":
    st.write("Upload ZIP containing photos + mapping.csv")
    zipf = st.file_uploader("ZIP", type=["zip"])
    if st.button("Upload") and zipf:
        ok, res = face_utils.register_bulk(zipf.read())
        if ok:
            st.dataframe(pd.DataFrame(res, columns=["roll", "ok", "msg"]))
        else:
            st.error(res)


# Attendance
if menu == "Attendance":
    img = st.camera_input("Capture photo")

    if img:
        pil = Image.open(io.BytesIO(img.getvalue()))
        enc = face_utils.get_encoding(pil)

        if enc is None:
            st.error("No face detected")
        else:
            roll, score = face_utils.match_face(enc)
            if roll:
                data = face_utils.load_encodings()
                name = data[roll]["name"]
                face_utils.mark_attendance(roll, name)
                st.success(f"Present: {name} ({roll})")
            else:
                st.warning("Not recognized")


# Students
if menu == "Students":
    data = face_utils.load_encodings()
    if not data:
        st.info("No students yet")
    else:
        rows = [{"roll": r, "name": d["name"], "photo": d["photo"]} for r, d in data.items()]
        st.dataframe(pd.DataFrame(rows))


# Attendance Records
if menu == "Records":
    files = glob.glob("attendance_*.csv")
    if files:
        file = st.selectbox("Select file", files)
        st.dataframe(pd.read_csv(file))
    else:
        st.info("No records yet")
