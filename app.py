import streamlit as st
from PIL import Image
import pandas as pd
import io, glob
import face_utils

st.set_page_config(page_title="AI Attendance System")
st.title("AI Attendance System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Register Single", "Register Bulk", "Take Attendance", "Students", "Records"]
)

if menu == "Home":
    st.write("Register students + Live camera attendance using face recognition.")


#  SINGLE STUDENT REGISTRATION
if menu == "Register Single":
    st.header("Single Student Register")

    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    photo = st.file_uploader("Upload JPG", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if not name or not roll or not photo:
            st.error("All fields required")
        else:
            img = Image.open(photo)
            ok, msg = face_utils.register_student(name, roll, img)
            st.success(msg) if ok else st.error(msg)


#  BULK REGISTRATION
if menu == "Register Bulk":
    st.header("Bulk Register (ZIP)")

    zip_file = st.file_uploader("Upload ZIP", type=["zip"])
    if st.button("Start Bulk Registration") and zip_file:
        ok, results = face_utils.register_bulk(zip_file.read())
        if ok:
            df = pd.DataFrame(results, columns=["roll", "ok", "msg"])
            st.dataframe(df)
        else:
            st.error(results)


#  TAKE ATTENDANCE
if menu == "Take Attendance":
    st.header("Live Attendance")

    img = st.camera_input("Capture photo")

    if img:
        pil = Image.open(io.BytesIO(img.getvalue()))
        enc = face_utils.image_to_encoding(pil)

        if enc is None:
            st.error("No face detected")
        else:
            roll, dist = face_utils.find_best_match(enc)

            if roll is None:
                st.warning("Not recognized. Marked absent.")
            else:
                data = face_utils.load_encodings()
                name = data[roll]["name"]
                st.success(f"Matched: {name} ({roll})")
                face_utils.mark_attendance(roll, name)


#  SHOW STUDENTS
if menu == "Students":
    st.header("Registered Students")
    data = face_utils.load_encodings()

    if not data:
        st.info("No students registered")
    else:
        df = pd.DataFrame(
            [{"roll": r, "name": v["name"], "photo": v["photo"]} for r, v in data.items()]
        )
        st.dataframe(df)

        for r, v in data.items():
            st.image(v["photo"], width=120, caption=f"{v['name']} ({r})")


#  ATTENDANCE RECORD
if menu == "Records":
    st.header("Attendance Files")

    files = glob.glob("attendance_*.csv")

    if not files:
        st.info("No attendance files yet")
    else:
        sel = st.selectbox("Select file", files)
        df = pd.read_csv(sel)
        st.dataframe(df)
