import streamlit as st
import requests  # Correct the import


def main():
    st.title("Fake News Classifier")
    user_input = st.text_input("Enter the text:")

    # JSON data to be sent to the API
    data = {"text": user_input}

    if st.button("Check"):
        if user_input:
            # Sending POST request to the Flask API
            response = requests.post("http://localhost:6000/predict", json=data)

            if response.status_code == 200:
                # Getting the prediction from the response
                prediction = response.json().get("prediction")

                if prediction == 0:
                    final_prediction = "Real"
                else:
                    final_prediction = "Fake"

                # Displaying the result
                st.success(f"The news is: {final_prediction}")
            else:
                st.error("Error while making the prediction.")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
