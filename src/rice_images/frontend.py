import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


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

    st.title("Image Classification")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            # Flatten the probabilities
            probabilities = result["probabilities"][0]

            # Show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # Create a dataframe for plotting the bar chart
            class_names = [
                "Arborio",
                "Basmati",
                "Ipsala",
                "Jasmine",
                "Karacadag",
            ]
            probability_data = {
                class_names[i]: probabilities[i]
                for i in range(len(class_names))
            }
            df = pd.DataFrame(
                list(probability_data.items()),
                columns=["Class", "Probability"],
            )

            # Display the bar chart with Class names on x-axis and Probability on y-axis
            st.bar_chart(df.set_index("Class")["Probability"])
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
