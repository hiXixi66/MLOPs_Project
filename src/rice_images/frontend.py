import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
import altair as alt


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/graphic-linker-448620-p3/locations/europe-west10"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)

    for service in services:
        if service.name.split("/")[-1] == "backendtest":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    print(f"Sending image with size: {len(image)} bytes")
    # Change the field name here from "image" to "file"
    response = requests.post(predict_url, files={"file": image}, timeout=60)

    print(f"Response Status Code: {response.status_code}")  # Log status code
    print(f"Response Content: {response.content}")  # Log content

    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Rice Image Classification")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = uploaded_file.read()
        with st.spinner("Classifying your image..."):
            result = classify_image(image, backend=backend)

        if result is not None:
            # Show the image and prediction
            with st.expander("View upload..."):
                st.image(image, caption="Uploaded Image")
            # Create a dataframe for plotting the bar chart
            class_names = [
                "Arborio",
                "Basmati",
                "Ipsala",
                "Jasmine",
                "Karacadag",
            ]
            colors = [
                "#BCB6FF",  # Arborio
                "#B8E1FF",  # Basmati
                "#BCE784",  # Ipsala
                "#5DD39E",  # Jasmine
                "#FFC09F",  # Karacadag
            ]

            st.success("Success!! 🎉")
            prediction = result["prediction"]
            # Flatten the probabilities
            probabilities = result["probabilities"][0]
            # Find the color for the predicted class
            class_color_map = dict(zip(class_names, colors))
            predicted_color = class_color_map.get(
                prediction, "#4CAF50"
            )  # Default color if not found

            font_style = "font-family: 'Poppins'; font-size: 24px;"

            # st.write("Prediction:", prediction)
            # Display the styled prediction
            st.markdown(
                f"<h3 style='text-align: center; {font_style}'>Prediction: "
                f"<span style='color: {predicted_color};'>"
                f"{prediction}</span></h3>",
                unsafe_allow_html=True,
            )

            # Prepare DataFrame
            probability_data = {
                "Class": class_names,
                "Probability": probabilities,
                "Color": colors,
            }
            df = pd.DataFrame(probability_data)

            # Create Altair chart
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    # Sort bars by class order
                    x=alt.X("Class", sort=class_names, title="Class"),
                    y=alt.Y("Probability", title="Probability"),
                    color=alt.Color("Color", scale=None),  # Use custom colors
                )
                .properties(width=600, height=400, title="Class Predictions")
            )

            with st.expander("Class Probabilities: Detailed View"):
                container = st.container(border=True)
                # Display the chart in Streamlit
                container.altair_chart(chart, use_container_width=True)

        else:
            st.error(
                "We encountered an issue. "
                "Please check your input or try later.",
                icon="🚨",
            )


if __name__ == "__main__":
    main()
