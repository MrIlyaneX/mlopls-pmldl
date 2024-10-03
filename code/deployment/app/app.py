# streamlit_app.py
import streamlit as st
import requests

from pydantic import ValidationError
from pydantic_models import StudentDataModel

import os
from dotenv import load_dotenv


load_dotenv()
FASTAPI_URL = os.getenv("FASTAPI_URL")


def create_input_fields(model):
    inputs = {}
    field_options = {
        "marital_status": [1, 2, 3, 4, 5, 6],
        "application_mode": [
            1,
            2,
            5,
            7,
            10,
            15,
            16,
            17,
            18,
            26,
            27,
            39,
            42,
            43,
            44,
            51,
            53,
            57,
        ],
        "application_order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "course": [
            33,
            171,
            8014,
            9003,
            9070,
            9085,
            9119,
            9130,
            9147,
            9238,
            9254,
            9500,
            9556,
            9670,
            9773,
            9853,
            9991,
        ],
        "daytime_evening_attendance": [1, 0],
        "previous_qualification": [
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            12,
            14,
            15,
            19,
            38,
            39,
            40,
            42,
            43,
        ],
        "previous_qualification_grade": range(0, 201),
        "nationality": [
            1,
            2,
            6,
            11,
            13,
            14,
            17,
            21,
            22,
            24,
            25,
            26,
            32,
            41,
            62,
            100,
            101,
            103,
            105,
            108,
            109,
        ],
        "mother_qualification": [
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            11,
            12,
            14,
            18,
            19,
            22,
            26,
            27,
            29,
            30,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ],
        "father_qualification": [
            1,
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            11,
            12,
            13,
            14,
            18,
            19,
            20,
            22,
            25,
            26,
            27,
            29,
            30,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ],
        "mother_occupation": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            90,
            99,
            101,
            102,
            103,
            121,
            122,
            123,
            124,
            131,
            132,
            134,
            135,
            141,
            143,
            144,
            151,
            152,
            153,
            154,
            161,
            163,
            171,
            172,
            174,
            175,
            181,
            182,
            183,
            192,
            193,
            194,
            195,
        ],
        "father_occupation": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            90,
            99,
            122,
            123,
            125,
            131,
            132,
            134,
            141,
            143,
            144,
            151,
            152,
            153,
            171,
            173,
            175,
            191,
            192,
            193,
            194,
        ],
        "displaced": [0, 1],
        "educational_special_needs": [0, 1],
        "debtor": [0, 1],
        "tuition_fees_up_to_date": [0, 1],
        "gender": [0, 1],
        "scholarship_holder": [0, 1],
        "age_at_enrollment": range(0, 201),
        "international": [0, 1],
    }

    for field_name, field_info in model.__annotations__.items():
        field_label = field_name.replace("_", " ").capitalize()

        if field_name in field_options.keys():
            inputs[field_name] = st.selectbox(field_label, field_options[field_name])
        elif field_info == int:
            inputs[field_name] = st.slider(field_label, format="%i", min_value=0)
        elif field_info == float:
            inputs[field_name] = st.number_input(
                field_label, format="%f", min_value=0.0
            )
        else:
            st.warning(f"Unsupported type for field '{field_name}'")

    return inputs


def main():
    st.title("Student Data for Prediction")

    inputs = create_input_fields(StudentDataModel)

    if st.button("Submit"):
        try:
            StudentDataModel(**inputs)
            response = requests.post(FASTAPI_URL, json=inputs)
            prediction = response.json()["prediction"]
        except ValidationError as e:
            st.error(f"Validation Error. Try again")


if __name__ == "__main__":
    main()
