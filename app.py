from flask import Flask, request, render_template_string, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
from collections import Counter
import os
import uuid

# ==========================================================
# FLASK SETUP
# ==========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ==========================================================
# LOAD YOUR TRAINED MODEL
# ==========================================================

model = YOLO("model/best.pt")


# ==========================================================
# POSSIBLE SOURCES
# ==========================================================

SOURCE_MAP = {
    "fiber": "Textile or synthetic-fiber sources",
    "film": "Plastic bags, wrappers and food packaging",
    "fragment": "Broken or weathered plastic products",
    "pellet": "Industrial plastic/resin pellet handling"
}


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

RECOMMENDATION_MAP = {
    "fiber": [
        "Check possible textile and synthetic-fiber sources around the sampling area.",
        "Improve filtration of wastewater or wash-water where applicable."
    ],

    "film": [
        "Investigate possible plastic packaging, wrappers or thin-film plastic sources.",
        "Improve plastic waste collection and filtration around the sampling area."
    ],

    "fragment": [
        "Inspect nearby plastic waste and degraded plastic products.",
        "Improve waste collection and filtration to reduce plastic contamination."
    ],

    "pellet": [
        "Check possible industrial plastic/resin pellet handling areas.",
        "Use proper containment and handling procedures to prevent pellet loss."
    ]
}


# ==========================================================
# RELATIVE SIZE FUNCTION
# ==========================================================

def get_relative_size(box_area, image_area):

    if image_area <= 0:
        return "Unknown"

    ratio = box_area / image_area

    if ratio < 0.005:
        return "Small"

    elif ratio < 0.02:
        return "Medium"

    else:
        return "Large"


# ==========================================================
# UPLOAD PAGE
# ==========================================================

UPLOAD_HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Microplastic Detection System</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #e9f7f5, #f4fbff);
    margin: 0;
    padding: 40px;
    color: #183b3b;
}

.container {
    max-width: 850px;
    margin: auto;
}

.card {
    background: white;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.10);
    text-align: center;
}

h1 {
    color: #087f70;
}

.subtitle {
    color: #666;
    margin-bottom: 30px;
}

.file-box {
    border: 2px dashed #087f70;
    padding: 30px;
    border-radius: 12px;
    margin: 25px 0;
    background: #f7fffd;
}

input[type=file] {
    width: 90%;
    padding: 12px;
}

button {
    background: #087f70;
    color: white;
    border: none;
    padding: 14px 30px;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #06665a;
}

.info {
    margin-top: 30px;
    padding: 20px;
    background: #f1f8f7;
    border-radius: 10px;
    text-align: left;
}

.info li {
    margin: 8px 0;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🔬 Microplastic Detection System</h1>

<p class="subtitle">
AI-Based Multiple Microscopic Water Image Analysis
</p>

<form method="POST" enctype="multipart/form-data">

<div class="file-box">

<h3>Select Microscopic Images</h3>

<input
    type="file"
    name="images"
    accept="image/*"
    multiple
    required
>

<p>
You can select multiple images at the same time.
</p>

</div>

<button type="submit">
🔍 Analyze All Images
</button>

</form>

<div class="info">

<h3>System Features</h3>

<ul>

<li>Multiple image analysis</li>

<li>Microplastic particle detection</li>

<li>Fiber, Film, Fragment and Pellet classification</li>

<li>Particle counting</li>

<li>Relative size estimation</li>

<li>Confidence score</li>

<li>Possible source estimation</li>

<li>Recommendations</li>

<li>Overall batch summary</li>

</ul>

</div>

</div>

</div>

</body>

</html>
"""


# ==========================================================
# RESULT PAGE
# ==========================================================

RESULT_HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Microplastic Analysis Results</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #eaf7f5, #f5fbff);
    margin: 0;
    padding: 30px;
    color: #183b3b;
}

.container {
    max-width: 1150px;
    margin: auto;
}

h1 {
    text-align: center;
    color: #087f70;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    margin-bottom: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.card h2 {
    color: #155e57;
}

.summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 15px;
}

.stat {
    background: #f1f8f7;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.stat h3 {
    margin: 0;
    font-size: 15px;
}

.stat p {
    font-size: 28px;
    font-weight: bold;
    color: #087f70;
    margin: 10px 0 0;
}

.confidence {
    text-align: center;
    background: #f4f8fa;
    padding: 25px;
    border-radius: 12px;
}

.confidence-number {
    font-size: 42px;
    font-weight: bold;
    color: #087f70;
}

.image-card {
    border: 1px solid #dce9e7;
    padding: 20px;
    margin-top: 25px;
    border-radius: 14px;
}

.image-title {
    color: #087f70;
}

.images {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}

.image-box {
    text-align: center;
}

.image-box img {
    width: 100%;
    max-height: 450px;
    object-fit: contain;
    border-radius: 12px;
    background: #f2f2f2;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

th {
    background: #e7f4f2;
    color: #155e57;
}

th, td {
    padding: 12px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

.source {
    background: #f1f8f7;
    border-left: 5px solid #3a9d8f;
    padding: 14px;
    margin: 10px 0;
    border-radius: 8px;
}

.recommendation {
    background: #f5faf8;
    border-left: 5px solid #5ca58e;
    padding: 14px;
    margin: 10px 0;
    border-radius: 8px;
}

.warning {
    background: #fff8df;
    border-left: 5px solid #e0ad28;
    padding: 15px;
    margin-top: 15px;
    border-radius: 8px;
    line-height: 1.6;
}

.scientific {
    background: #eef6ff;
    border-left: 5px solid #4682b4;
    padding: 20px;
    border-radius: 10px;
    line-height: 1.7;
}

.button-area {
    text-align: center;
    margin: 30px 0;
}

.button {
    display: inline-block;
    background: #087f70;
    color: white;
    text-decoration: none;
    padding: 13px 25px;
    border-radius: 8px;
}

</style>

</head>

<body>

<div class="container">

<h1>🔬 Microplastic Analysis Results</h1>


<!-- =====================================================
OVERALL SUMMARY
===================================================== -->

<div class="card">

<h2>📊 Overall Batch Summary</h2>

<div class="summary">

<div class="stat">
<h3>Images Analyzed</h3>
<p>{{ number_of_images }}</p>
</div>

<div class="stat">
<h3>Total Particles</h3>
<p>{{ total_particles }}</p>
</div>

<div class="stat">
<h3>Fiber</h3>
<p>{{ total_counts["fiber"] }}</p>
</div>

<div class="stat">
<h3>Film</h3>
<p>{{ total_counts["film"] }}</p>
</div>

<div class="stat">
<h3>Fragment</h3>
<p>{{ total_counts["fragment"] }}</p>
</div>

<div class="stat">
<h3>Pellet</h3>
<p>{{ total_counts["pellet"] }}</p>
</div>

</div>

</div>


<!-- =====================================================
OVERALL CONFIDENCE
===================================================== -->

<div class="card">

<h2>🎯 Overall Model Confidence</h2>

<div class="confidence">

<div class="confidence-number">
{{ overall_confidence }}%
</div>

<h3>
Overall Detection Level: {{ overall_level }}
</h3>

<p>
This represents the average confidence of the detected
particles across all uploaded images.
</p>

<div class="warning">

<strong>Important:</strong>

Confidence is not the same as model accuracy.
Results should be reviewed using the detection visualization.

</div>

</div>

</div>


<!-- =====================================================
INDIVIDUAL IMAGE RESULTS
===================================================== -->

<div class="card">

<h2>🖼️ Individual Image Results</h2>


{% for image in image_results %}

<div class="image-card">

<h2 class="image-title">
Image {{ image.number }}
</h2>

<p>
<strong>File:</strong> {{ image.filename }}
</p>


<div class="images">

<div class="image-box">

<h3>Original Water Sample</h3>

<img src="{{ image.original_url }}">

</div>


<div class="image-box">

<h3>AI Detection Result</h3>

<img src="{{ image.result_url }}">

</div>

</div>


<!-- IMAGE COUNTS -->

<h3>📊 Particle Count</h3>

<div class="summary">

<div class="stat">
<h3>Total</h3>
<p>{{ image.total }}</p>
</div>

<div class="stat">
<h3>Fiber</h3>
<p>{{ image.counts.get("fiber", 0) }}</p>
</div>

<div class="stat">
<h3>Film</h3>
<p>{{ image.counts.get("film", 0) }}</p>
</div>

<div class="stat">
<h3>Fragment</h3>
<p>{{ image.counts.get("fragment", 0) }}</p>
</div>

<div class="stat">
<h3>Pellet</h3>
<p>{{ image.counts.get("pellet", 0) }}</p>
</div>

</div>


<!-- PARTICLE DETAILS -->

<h3>🔍 Particle Details</h3>

{% if image.detections %}

<table>

<tr>
<th>Particle Type</th>
<th>Confidence</th>
<th>Relative Size</th>
</tr>

{% for particle in image.detections %}

<tr>

<td>{{ particle.type }}</td>

<td>{{ particle.confidence }}%</td>

<td>{{ particle.size }}</td>

</tr>

{% endfor %}

</table>

{% else %}

<p>
No particles detected in this image.
</p>

{% endif %}


<!-- IMAGE CONFIDENCE -->

<h3>🎯 Image Confidence</h3>

<p>
<strong>{{ image.confidence }}%</strong>
average detection confidence.
</p>


<!-- SOURCES -->

<h3>🌊 Possible Sources</h3>

{% if image.sources %}

{% for source in image.sources %}

<div class="source">

<strong>{{ source.type }}:</strong>
{{ source.source }}

</div>

{% endfor %}

{% else %}

<p>
No possible source estimate available.
</p>

{% endif %}


</div>

{% endfor %}

</div>


<!-- =====================================================
COMBINED SOURCES
===================================================== -->

<div class="card">

<h2>🌊 Combined Possible Sources</h2>

{% for source in combined_sources %}

<div class="source">

<strong>{{ source.type }}:</strong>
{{ source.source }}

</div>

{% endfor %}

<div class="warning">

<strong>⚠️ Important:</strong>

These are possible sources estimated from particle
morphology/type. They are not confirmed pollution sources.

</div>

</div>


<!-- =====================================================
RECOMMENDATIONS
===================================================== -->

<div class="card">

<h2>💡 Recommendations</h2>

{% for recommendation in combined_recommendations %}

<div class="recommendation">

✓ {{ recommendation }}

</div>

{% endfor %}

</div>


<!-- =====================================================
SCIENTIFIC NOTE
===================================================== -->

<div class="card">

<h2>🔬 Scientific Note</h2>

<div class="scientific">

<p>
This system performs
<strong>image-based screening of suspected microplastic particles</strong>
from microscopic water-sample images.
</p>

<p>
Visual detection alone cannot definitively confirm that a detected
particle is plastic.
</p>

<p>
For laboratory confirmation of polymer type,
techniques such as <strong>FTIR</strong> or
<strong>Raman spectroscopy</strong> can be considered.
</p>

<p>
<strong>Relative size</strong> is estimated from the image bounding box.
Actual particle size in micrometers (µm) requires microscope calibration
or a scale reference.
</p>

</div>

</div>


<!-- =====================================================
BACK BUTTON
===================================================== -->

<div class="button-area">

<a href="/" class="button">
🔄 Analyze More Images
</a>

</div>


</div>

</body>

</html>
"""


# ==========================================================
# HOME + IMAGE ANALYSIS
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # ------------------------------------------------------
    # SHOW UPLOAD PAGE
    # ------------------------------------------------------

    if request.method == "GET":

        return render_template_string(UPLOAD_HTML)


    # ------------------------------------------------------
    # GET MULTIPLE IMAGES
    # ------------------------------------------------------

    files = request.files.getlist("images")


    # Remove empty selections

    files = [
        file
        for file in files
        if file and file.filename
    ]


    if not files:

        return """
        <h2>No images were selected.</h2>
        <a href="/">Go Back</a>
        """


    # ------------------------------------------------------
    # VARIABLES
    # ------------------------------------------------------

    total_counts = Counter()

    total_particles = 0

    all_confidences = []

    image_results = []


    # ------------------------------------------------------
    # PROCESS EACH IMAGE
    # ------------------------------------------------------

    for image_number, file in enumerate(files, start=1):

        # --------------------------------------------------
        # CREATE UNIQUE FILE NAME
        # --------------------------------------------------

        original_filename = secure_filename(
            file.filename
        )

        unique_name = (
            str(uuid.uuid4())[:8]
            + "_"
            + original_filename
        )


        upload_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )


        file.save(upload_path)


        # --------------------------------------------------
        # RUN YOLO
        # --------------------------------------------------

        results = model(
            upload_path,
            conf=0.25
        )


        result = results[0]


        # --------------------------------------------------
        # CREATE DETECTION IMAGE
        # --------------------------------------------------

        annotated = result.plot()


        result_filename = (
            "detected_"
            + unique_name
        )


        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )


        # YOLO plot image is BGR.
        # Convert BGR -> RGB before saving with PIL.

        annotated_rgb = annotated[:, :, ::-1]

        Image.fromarray(
            annotated_rgb
        ).save(result_path)


        # --------------------------------------------------
        # IMAGE SIZE
        # --------------------------------------------------

        original_image = Image.open(
            upload_path
        )

        width, height = original_image.size

        image_area = width * height


        # --------------------------------------------------
        # DETECTION VARIABLES
        # --------------------------------------------------

        detections = []

        image_counts = Counter()

        image_confidences = []

        detected_types = []


        # --------------------------------------------------
        # CHECK BOXES
        # --------------------------------------------------

        if result.boxes is not None:

            number_of_boxes = len(
                result.boxes
            )


            for i in range(number_of_boxes):

                # ------------------------------------------
                # CLASS
                # ------------------------------------------

                class_id = int(
                    result.boxes.cls[i]
                )


                class_name = str(
                    result.names[class_id]
                ).lower()


                # ------------------------------------------
                # CONFIDENCE
                # ------------------------------------------

                confidence = float(
                    result.boxes.conf[i]
                )


                confidence_percent = round(
                    confidence * 100,
                    2
                )


                # ------------------------------------------
                # COUNT
                # ------------------------------------------

                image_counts[class_name] += 1

                total_counts[class_name] += 1

                detected_types.append(
                    class_name
                )


                # ------------------------------------------
                # CONFIDENCE LIST
                # ------------------------------------------

                image_confidences.append(
                    confidence_percent
                )

                all_confidences.append(
                    confidence_percent
                )


                # ------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------

                box = result.boxes.xyxy[i].tolist()

                x1, y1, x2, y2 = box

                box_width = max(
                    0,
                    x2 - x1
                )

                box_height = max(
                    0,
                    y2 - y1
                )

                box_area = (
                    box_width *
                    box_height
                )


                # ------------------------------------------
                # RELATIVE SIZE
                # ------------------------------------------

                size = get_relative_size(
                    box_area,
                    image_area
                )


                # ------------------------------------------
                # SAVE PARTICLE
                # ------------------------------------------

                detections.append({

                    "type":
                        class_name.title(),

                    "confidence":
                        confidence_percent,

                    "size":
                        size

                })


        # --------------------------------------------------
        # IMAGE TOTAL
        # --------------------------------------------------

        image_total = len(detections)

        total_particles += image_total


        # --------------------------------------------------
        # IMAGE AVERAGE CONFIDENCE
        # --------------------------------------------------

        if image_confidences:

            image_confidence = round(
                sum(image_confidences)
                / len(image_confidences),
                2
            )

        else:

            image_confidence = 0


        # --------------------------------------------------
        # UNIQUE PARTICLE TYPES
        # --------------------------------------------------

        unique_types = list(
            dict.fromkeys(
                detected_types
            )
        )


        # --------------------------------------------------
        # SOURCES
        # --------------------------------------------------

        image_sources = []

        for particle_type in unique_types:

            if particle_type in SOURCE_MAP:

                image_sources.append({

                    "type":
                        particle_type.title(),

                    "source":
                        SOURCE_MAP[
                            particle_type
                        ]

                })


        # --------------------------------------------------
        # IMAGE RECOMMENDATIONS
        # --------------------------------------------------

        image_recommendations = []

        for particle_type in unique_types:

            if particle_type in RECOMMENDATION_MAP:

                for recommendation in RECOMMENDATION_MAP[
                    particle_type
                ]:

                    if recommendation not in image_recommendations:

                        image_recommendations.append(
                            recommendation
                        )


        # --------------------------------------------------
        # IMAGE URLS
        # --------------------------------------------------

        original_url = url_for(
            "static",
            filename="uploads/" + unique_name
        )


        result_url = url_for(
            "static",
            filename="results/" + result_filename
        )


        # --------------------------------------------------
        # STORE IMAGE RESULT
        # --------------------------------------------------

        image_results.append({

            "number":
                image_number,

            "filename":
                original_filename,

            "original_url":
                original_url,

            "result_url":
                result_url,

            "detections":
                detections,

            "counts":
                dict(image_counts),

            "total":
                image_total,

            "confidence":
                image_confidence,

            "sources":
                image_sources,

            "recommendations":
                image_recommendations

        })


    # ======================================================
    # OVERALL CONFIDENCE
    # ======================================================

    if all_confidences:

        overall_confidence = round(
            sum(all_confidences)
            / len(all_confidences),
            2
        )

    else:

        overall_confidence = 0


    # ======================================================
    # OVERALL LEVEL
    # ======================================================

    if overall_confidence >= 80:

        overall_level = "High"

    elif overall_confidence >= 60:

        overall_level = "Moderate"

    else:

        overall_level = "Low"


    # ======================================================
    # MAKE SURE ALL FOUR CLASSES EXIST
    # ======================================================

    final_counts = {

        "fiber":
            total_counts.get(
                "fiber",
                0
            ),

        "film":
            total_counts.get(
                "film",
                0
            ),

        "fragment":
            total_counts.get(
                "fragment",
                0
            ),

        "pellet":
            total_counts.get(
                "pellet",
                0
            )
    }


    # ======================================================
    # COMBINED POSSIBLE SOURCES
    # ======================================================

    combined_sources = []

    for particle_type in [
        "fiber",
        "film",
        "fragment",
        "pellet"
    ]:

        if final_counts[particle_type] > 0:

            combined_sources.append({

                "type":
                    particle_type.title(),

                "source":
                    SOURCE_MAP[
                        particle_type
                    ]

            })


    # ======================================================
    # COMBINED RECOMMENDATIONS
    # ======================================================

    combined_recommendations = [

        "Use proper filtration and clean sample-handling procedures to reduce contamination.",

        "Analyze multiple microscopic images from the same water sample for more representative results."

    ]


    for particle_type in [
        "fiber",
        "film",
        "fragment",
        "pellet"
    ]:

        if final_counts[particle_type] > 0:

            for recommendation in RECOMMENDATION_MAP[
                particle_type
            ]:

                if recommendation not in combined_recommendations:

                    combined_recommendations.append(
                        recommendation
                    )


    combined_recommendations.append(
        "For laboratory confirmation of polymer type, techniques such as FTIR or Raman spectroscopy can be considered."
    )


    # ======================================================
    # SHOW RESULT PAGE
    # ======================================================

    return render_template_string(

        RESULT_HTML,

        image_results=image_results,

        number_of_images=len(files),

        total_particles=total_particles,

        total_counts=final_counts,

        overall_confidence=overall_confidence,

        overall_level=overall_level,

        combined_sources=combined_sources,

        combined_recommendations=combined_recommendations

    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )