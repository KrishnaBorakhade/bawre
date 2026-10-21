# Pill Detection System

## Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)

* [Features](https://www.google.com/search?q=%23features)

* [Installation](https://www.google.com/search?q=%23installation)

* [Usage](https://www.google.com/search?q=%23usage)

* [Project Structure](https://www.google.com/search?q=%23project-structure)

* [Technologies Used](https://www.google.com/search?q=%23technologies-used)

* [Contributing](https://www.google.com/search?q=%23contributing)

* [License](https://www.google.com/search?q=%23license)

## Project Overview

The Pill Detection System is a web-based application designed to identify pills using image recognition and provide information about them, including potential side effects. It leverages a Flask backend for the web interface and a pre-trained machine learning model (MobileNet) for image classification. This system aims to assist users in quickly identifying unknown pills and accessing relevant drug information.

## Features

* **Pill Image Recognition**: Upload an image of a pill to identify it.

* **Drug Information Retrieval**: Access details about the identified pill.

* **Side Effects Information**: View potential side effects associated with the drug.

* **User-Friendly Web Interface**: Intuitive design for easy interaction.

## Installation

To set up and run this project locally, follow these steps:

### Prerequisites

* Python 3.x

* pip (Python package installer)

### Steps

1. **Clone the repository (if applicable):**


git clone <repository_url>
cd pill-detection-system # Replace with your actual project directory name


*(Note: If you don't have a Git repository, ensure you are in the root directory of your project where `flask-app` is located.)*

2. **Navigate to the Flask application directory:**


cd flask-app


3. **Create a virtual environment (recommended):**


python -m venv venv


4. **Activate the virtual environment:**

* **On Windows:**

  ```
  .\venv\Scripts\activate
  ```

* **On macOS/Linux:**

  ```
  source venv/bin/activate
  ```

5. **Install the required Python packages:**


pip install -r requirements.txt


6. **Initialize the database:**


python create_db.py


This script will set up the `drugs.db` SQLite database.

## Usage

1. **Start the Flask application:**
Ensure your virtual environment is activated and you are in the `flask-app` directory.


python app.py


2. **Access the application:**
Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address displayed in your terminal).

3. **Upload a pill image:**
Use the provided interface to upload an image of a pill for identification. The system will process the image and display relevant drug information and side effects.

## Project Structure


.
├── flask-app/
│   ├── pycache/            # Python compiled bytecode
│   ├── dataset/                # Dataset for training/testing the model (e.g., pill images)
│   ├── instance/               # Flask instance folder (e.g., configuration, database)
│   ├── models/                 # Python modules related to ML model handling
│   ├── static/                 # Static files (CSS, JS, images like celecoxib.jpg, img.png)
│   │   ├── celecoxib.jpg       # Example pill image
│   │   └── img.png             # Another example image
│   ├── templates/              # Jinja2 HTML templates for the web interface
│   ├── app.py                  # Main Flask application file
│   ├── create_db.py            # Script to initialize and populate the SQLite database
│   ├── drug_side_effects.py    # Logic for handling drug side effects data
│   ├── drugs.db                # SQLite database file (contains drug information)
│   ├── main.py                 # Potentially another entry point or script (e.g., for model training/testing)
│   ├── MobileNet.keras         # The trained Keras model for pill detection
│   ├── models.py               # Python script for defining or loading ML models
│   └── requirements.txt        # List of Python dependencies


## Technologies Used

* **Python**: The core programming language.

* **Flask**: Web framework for building the application.

* **Keras / TensorFlow**: For building and utilizing the deep learning model (`MobileNet.keras`).

* **SQLite**: Database for storing drug information.

* **HTML/CSS/JavaScript**: For the front-end web interface.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
