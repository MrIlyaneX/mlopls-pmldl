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
    for field_name, field_info in model.__annotations__.items():
        field_label = field_name.replace("_", " ").capitalize()
        
        if field_info == int:
            inputs[field_name] = st.slider(
                field_label,
                format="%i",
                min_value=0
            )
        elif field_info == float:
            inputs[field_name] = st.number_input(
                field_label,
                format="%f",
                min_value=0.0
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
