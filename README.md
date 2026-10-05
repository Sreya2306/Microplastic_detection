🔬 AI-Based Microplastic Detection System

An AI-powered computer vision project that detects and analyzes suspected microplastic particles in microscopic water-sample images using a trained YOLO11 segmentation model.

🚀 Features

- Upload multiple microscopic images at the same time.
- Detect and classify suspected microplastic particles.
- Identify four particle types: Fiber, Film, Fragment, and Pellet.
- Count particles in each image and calculate combined totals.
- Display AI detection visualizations.
- Show confidence scores for detected particles.
- Estimate relative particle size.
- Suggest possible pollution sources.
- Provide recommendations based on detected particle types.
- Generate an overall summary for multiple images.

🧠 Technologies Used

- Python — Programming language
- YOLO11 Segmentation — Particle detection and segmentation
- Ultralytics — AI model implementation
- Flask — Web application framework
- HTML and CSS — User interface
- Pillow — Image processing
- Roboflow — Dataset preparation
- GitHub — Version control

⚙️ How It Works

1. The user uploads one or more microscopic water images.
2. The trained YOLO model analyzes each image.
3. The system detects and classifies suspected particles.
4. It calculates particle counts, confidence scores, and relative sizes.
5. It estimates possible sources based on particle types.
6. It generates recommendations and a combined analysis summary.

🧩 Particle Classification

Particle Type| Possible Source
Fiber| Textiles or synthetic-fiber materials
Film| Plastic bags, wrappers, or packaging
Fragment| Broken or weathered plastic products
Pellet| Industrial plastic or resin pellets

These sources are estimates based on particle appearance, not confirmed pollution sources.

📊 Output

For each uploaded image, the system displays:

- Original microscopic image
- AI detection result
- Total detected particles
- Particle classification and counts
- Individual detection confidence
- Relative particle size
- Possible source estimates

The combined report includes total images analyzed, overall particle counts, possible sources, and recommendations.

📏 Relative Size Estimation

The system categorizes detected particles as Small, Medium, or Large based on their bounding-box area relative to the image.

Actual particle size in micrometers (µm) requires microscope calibration or a known scale reference.

🌊 Recommendations

Depending on the detected particle types, the system may recommend:

- Improving water filtration where appropriate.
- Investigating possible textile and synthetic-fiber sources.
- Checking plastic packaging and nearby plastic waste.
- Improving sample handling to reduce contamination.
- Analyzing additional images from the same water sample.
- Considering laboratory confirmation of suspected microplastics.

📂 Project Structure

microplastic_detection/
├── dataset/
├── model/
│   └── best.pt
├── static/
│   ├── uploads/
│   └── results/
├── test_images/
├── app.py
├── train.py
├── detect.py
├── requirements.txt
└── README.md

📦 Dataset

The project uses a microscopic microplastic dataset with four classes: Fiber, Film, Fragment, and Pellet.

Dataset: "Microplastic Dataset on Roboflow Universe" (https://universe.roboflow.com/project-aunby/microplastic-final-kpdl3/dataset/2)

Dataset license: CC BY 4.0. Follow the dataset license and attribution requirements when redistributing dataset material.

🤖 Trained Model

The trained model is stored at:

model/best.pt

The model performs image-based particle detection, classification, and segmentation.

💻 Installation

1. Clone the repository

Replace the URL with your actual GitHub repository URL.

git clone https://github.com/tanujadesai1122/microplastic_detection
cd microplastic_detection

2. Create a virtual environment

python -m venv venv

3. Activate it on Windows

venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

If you need to create "requirements.txt", add:

flask
ultralytics
pillow
opencv-python
werkzeug

5. Run the application

python app.py

Open the following address in your browser:

http://127.0.0.1:5000

🖼️ How to Use

1. Open the application in your browser.
2. Select multiple microscopic water images.
3. Click Analyze All Images.
4. Wait for the model to process the images.
5. Review the individual results and combined summary.

🔬 Scientific Limitations

This project performs image-based screening of suspected microplastic particles. Visual appearance alone cannot definitively prove that a particle is plastic or establish its pollution source.

For laboratory-level polymer identification, techniques such as FTIR spectroscopy or Raman spectroscopy may be considered.

The confidence score is not the same as model accuracy. Results should be reviewed, and scientific conclusions should be confirmed using appropriate laboratory methods.

🌱 Applications and Impact

Potential applications include:

- Preliminary environmental screening
- Water-quality research
- Academic and educational projects
- Microscopic image analysis
- Environmental pollution studies

The project aims to make preliminary microscopic image analysis easier and faster.

🔮 Future Scope

- Calibrated particle-size measurement
- Automatic PDF reports
- Graphs and statistical summaries
- Database for sample history
- Water-sample location tracking
- Historical sample comparisons
- Integration with laboratory analysis
- Cloud deployment
- Improved detection through additional training data

⚠️ Disclaimer

This project is intended for educational, research, and preliminary screening purposes. Its predictions should not be treated as definitive confirmation of microplastic composition or pollution source.

---

Project: AI-Based Microplastic Detection System
Domain: Artificial Intelligence and Machine Learning
Application: Preliminary Microplastic Screening in Water Samples
Model: YOLO11 Segmentation
Framework: Flask
Language: Python