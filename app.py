from flask import Flask, request, render_template_string, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
import os
import uuid

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = YOLO("model/best.pt")


# =========================================================
# POSSIBLE SOURCES
# =========================================================

SOURCE_MAP = {
    "fiber": "Textile or synthetic-fiber sources",
    "film": "Plastic bags, wrappers and food packaging",
    "fragment": "Broken or weathered plastic products",
    "pellet": "Industrial plastic/resin pellet handling"
}


# =========================================================
# RECOMMENDATIONS
# =========================================================

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


# =========================================================
# SIZE CLASSIFICATION
# =========================================================

def calculate_relative_size(box_area, image_area):

    ratio = box_area / image_area

    if ratio < 0.005:
        return "Small"

    elif ratio < 0.02:
        return "Medium"

    else:
        return "Large"


# =========================================================
# MAIN PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    # -----------------------------------------------------
    # UPLOAD PAGE
    # -----------------------------------------------------

    if request.method == "GET":

        return render_template_string("""
<!DOCTYPE html>

<html>

<head>

<title>Microplastic Detection System</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #e8f7f5, #f4fbff);
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

input[type=file] {
    padding: 15px;
    border: 2px dashed #75bdb3;
    border-radius: 10px;
    width: 80%;
    margin: 20px;
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
    background: #f1f8f7;
    padding: 18px;
    border-radius: 10px;
    text-align: left;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🔬 Microplastic Detection System</h1>

<p class="subtitle">
AI-Based Microscopic Image Analysis
</p>

<form method="POST" enctype="multipart/form-data">

<input
    type="file"
    name="image"
    accept="image/*"
    required
>

<br>

<button type="submit">
Analyze Water Sample
</button>

</form>

<div class="info">

<h3>System Features</h3>

<ul>
<li>Microplastic particle detection</li>
<li>Particle type classification</li>
<li>Particle counting</li>
<li>Relative size estimation</li>
<li>Detection confidence</li>
<li>Possible source estimation</li>
<li>Recommendations</li>
</ul>

</div>

</div>

</div>

</body>

</html>
        """)


    # =====================================================
    # GET UPLOADED FILE
    # =====================================================

    file = request.files.get("image")

    if not file or file.filename == "":
        return "Please select an image."


    # =====================================================
    # SAVE ORIGINAL IMAGE
    # =====================================================

    filename = secure_filename(file.filename)

    unique_name = str(uuid.uuid4())[:8] + "_" + filename

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(upload_path)


    # =====================================================
    # RUN YOLO MODEL
    # =====================================================

    results = model(
        upload_path,
        conf=0.25
    )

    result = results[0]


    # =====================================================
    # CREATE DETECTION IMAGE
    # =====================================================

    annotated_image = result.plot()

    result_filename = "detected_" + unique_name

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    # YOLO plot returns BGR image
    Image.fromarray(
        annotated_image[..., ::-1]
    ).save(result_path)


    # =====================================================
    # IMAGE SIZE
    # =====================================================

    original_image = Image.open(upload_path)

    image_width, image_height = original_image.size

    image_area = image_width * image_height


    # =====================================================
    # DETECTION INFORMATION
    # =====================================================

    detections = []

    counts = {
        "fiber": 0,
        "film": 0,
        "fragment": 0,
        "pellet": 0
    }

    detected_types = []

    confidence_values = []


    # =====================================================
    # READ DETECTIONS
    # =====================================================

    if result.boxes is not None:

        for i in range(len(result.boxes)):

            class_id = int(
                result.boxes.cls[i]
            )

            confidence = float(
                result.boxes.conf[i]
            )

            class_name = result.names[class_id].lower()

            confidence_percent = round(
                confidence * 100,
                2
            )

            confidence_values.append(
                confidence_percent
            )


            # ---------------------------------------------
            # COUNT PARTICLES
            # ---------------------------------------------

            if class_name in counts:
                counts[class_name] += 1

            detected_types.append(class_name)


            # ---------------------------------------------
            # BOUNDING BOX
            # ---------------------------------------------

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


            # ---------------------------------------------
            # RELATIVE SIZE
            # ---------------------------------------------

            relative_size = calculate_relative_size(
                box_area,
                image_area
            )


            # ---------------------------------------------
            # STORE PARTICLE
            # ---------------------------------------------

            detections.append({

                "type": class_name.title(),

                "confidence": confidence_percent,

                "size": relative_size

            })


    # =====================================================
    # TOTAL PARTICLES
    # =====================================================

    total_particles = len(detections)


    # =====================================================
    # AVERAGE CONFIDENCE
    # =====================================================

    if confidence_values:

        average_confidence = round(
            sum(confidence_values)
            /
            len(confidence_values),
            2
        )

    else:

        average_confidence = 0


    # =====================================================
    # DETECTION LEVEL
    # =====================================================

    if average_confidence >= 80:

        detection_level = "High"

    elif average_confidence >= 60:

        detection_level = "Moderate"

    else:

        detection_level = "Low"


    # =====================================================
    # REMOVE DUPLICATE TYPES
    # =====================================================

    detected_types = list(
        dict.fromkeys(
            detected_types
        )
    )


    # =====================================================
    # POSSIBLE SOURCES
    # =====================================================

    possible_sources = []

    for particle_type in detected_types:

        if particle_type in SOURCE_MAP:

            possible_sources.append({

                "type": particle_type.title(),

                "source": SOURCE_MAP[
                    particle_type
                ]

            })


    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    recommendations = []

    # General recommendation
    recommendations.append(
        "Use proper filtration and clean sample-handling procedures to reduce contamination."
    )

    recommendations.append(
        "Repeat the analysis with additional microscopic images from the same water sample."
    )


    # Type-specific recommendations

    for particle_type in detected_types:

        if particle_type in RECOMMENDATION_MAP:

            for recommendation in RECOMMENDATION_MAP[
                particle_type
            ]:

                if recommendation not in recommendations:

                    recommendations.append(
                        recommendation
                    )


    # Scientific recommendation

    recommendations.append(
        "For laboratory confirmation of polymer type, techniques such as FTIR or Raman spectroscopy can be considered."
    )


    # =====================================================
    # IMAGE URLS
    # =====================================================

    original_url = url_for(
        "static",
        filename="uploads/" + unique_name
    )

    result_url = url_for(
        "static",
        filename="results/" + result_filename
    )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template_string("""

<!DOCTYPE html>

<html>

<head>

<title>Microplastic Analysis Results</title>

<style>

body {

    font-family: Arial, sans-serif;

    background: linear-gradient(
        135deg,
        #eaf7f5,
        #f5fbff
    );

    margin: 0;

    padding: 30px;

    color: #183b3b;
}


.container {

    max-width: 1100px;

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

    box-shadow:
        0 4px 15px
        rgba(0,0,0,0.08);
}


.card h2 {

    margin-top: 0;

    color: #155e57;

}


.summary {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(160px, 1fr)
        );

    gap: 15px;
}


.stat {

    background: #f1f8f7;

    padding: 20px;

    border-radius: 12px;

    text-align: center;
}


.stat h3 {

    margin: 0;

    font-size: 17px;

}


.stat p {

    font-size: 30px;

    font-weight: bold;

    color: #087f70;

    margin: 10px 0 0;
}


.images {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(300px, 1fr)
        );

    gap: 20px;
}


.image-box {

    text-align: center;
}


.image-box img {

    width: 100%;

    border-radius: 12px;

    box-shadow:
        0 3px 10px
        rgba(0,0,0,0.12);
}


table {

    width: 100%;

    border-collapse: collapse;
}


th {

    background: #e7f4f2;

    color: #155e57;
}


th, td {

    padding: 13px;

    border-bottom:
        1px solid #ddd;

    text-align: left;
}


.source-item {

    background: #f1f8f7;

    border-left:
        5px solid #3a9d8f;

    padding: 15px;

    margin-bottom: 10px;

    border-radius: 8px;
}


.recommendation {

    background: #f5faf8;

    border-left:
        5px solid #5ca58e;

    padding: 14px;

    margin-bottom: 10px;

    border-radius: 8px;
}


.confidence-box {

    text-align: center;

    padding: 25px;

    background: #f4f8fa;

    border-radius: 12px;
}


.confidence-number {

    font-size: 42px;

    font-weight: bold;

    color: #087f70;
}


.confidence-level {

    font-size: 20px;

    font-weight: bold;
}


.confidence-note {

    margin-top: 15px;

    padding: 15px;

    background: #fff8df;

    border-left:
        5px solid #e0ad28;

    border-radius: 8px;

    text-align: left;

    line-height: 1.6;
}


.warning {

    margin-top: 18px;

    padding: 15px;

    background: #fff8df;

    border-left:
        5px solid #e0ad28;

    border-radius: 8px;

    line-height: 1.6;
}


.scientific-note {

    background: #eef6ff;

    border-left:
        5px solid #4682b4;

    padding: 20px;

    border-radius: 10px;

    line-height: 1.7;
}


.button {

    display: inline-block;

    background: #087f70;

    color: white;

    text-decoration: none;

    padding: 13px 25px;

    border-radius: 8px;

    margin-top: 10px;
}


.button:hover {

    background: #06665a;
}


.center {

    text-align: center;
}


.empty {

    padding: 15px;

    background: #f5f5f5;

    border-radius: 8px;
}


</style>

</head>


<body>


<div class="container">


<h1>
🔬 Microplastic Analysis Results
</h1>


<!-- ================================================= -->
<!-- SUMMARY -->
<!-- ================================================= -->

<div class="card">

<h2>📊 Detection Summary</h2>

<div class="summary">


<div class="stat">

<h3>Fiber</h3>

<p>
{{ counts["fiber"] }}
</p>

</div>


<div class="stat">

<h3>Fragment</h3>

<p>
{{ counts["fragment"] }}
</p>

</div>


<div class="stat">

<h3>Film</h3>

<p>
{{ counts["film"] }}
</p>

</div>


<div class="stat">

<h3>Pellet</h3>

<p>
{{ counts["pellet"] }}
</p>

</div>


<div class="stat">

<h3>Total</h3>

<p>
{{ total_particles }}
</p>

</div>


</div>

</div>


<!-- ================================================= -->
<!-- IMAGES -->
<!-- ================================================= -->

<div class="card">

<h2>🖼️ Sample Visualization</h2>

<div class="images">


<div class="image-box">

<h3>Original Water Sample</h3>

<img src="{{ original_url }}">

</div>


<div class="image-box">

<h3>AI Detection Result</h3>

<img src="{{ result_url }}">

</div>


</div>

</div>


<!-- ================================================= -->
<!-- PARTICLE DETAILS -->
<!-- ================================================= -->

<div class="card">

<h2>🔍 Particle Details</h2>


{% if detections %}

<table>

<tr>

<th>Particle Type</th>

<th>Confidence</th>

<th>Relative Size</th>

</tr>


{% for particle in detections %}

<tr>

<td>
{{ particle["type"] }}
</td>

<td>
{{ particle["confidence"] }}%
</td>

<td>
{{ particle["size"] }}
</td>

</tr>

{% endfor %}

</table>


{% else %}

<div class="empty">

No particles were detected in this image.

</div>

{% endif %}

</div>


<!-- ================================================= -->
<!-- CONFIDENCE -->
<!-- ================================================= -->

<div class="card">

<h2>🎯 Model Confidence</h2>


<div class="confidence-box">

<div class="confidence-number">

{{ average_confidence }}%

</div>


<div class="confidence-level">

Overall Detection Level:
{{ detection_level }}

</div>


<div class="confidence-note">

<strong>Interpretation:</strong>

The confidence represents the average confidence
of the detected particles in this image.

Results should be reviewed using the detection
visualization and, where possible, additional
microscopic images from the same sample.

</div>

</div>

</div>


<!-- ================================================= -->
<!-- POSSIBLE SOURCE -->
<!-- ================================================= -->

<div class="card">

<h2>🌊 Possible Sources</h2>


{% if possible_sources %}


{% for item in possible_sources %}

<div class="source-item">

<strong>
{{ item["type"] }}:
</strong>

{{ item["source"] }}

</div>

{% endfor %}


<div class="warning">

⚠️ <strong>Important:</strong>

These are possible sources estimated from
particle morphology/type.

They are <strong>not confirmed pollution sources</strong>.

</div>


{% else %}


<div class="empty">

No possible source estimate is available.

</div>


{% endif %}

</div>


<!-- ================================================= -->
<!-- RECOMMENDATIONS -->
<!-- ================================================= -->

<div class="card">

<h2>💡 Recommendations</h2>


{% for recommendation in recommendations %}

<div class="recommendation">

✓ {{ recommendation }}

</div>

{% endfor %}


</div>


<!-- ================================================= -->
<!-- SCIENTIFIC NOTE -->
<!-- ================================================= -->

<div class="card">

<h2>🔬 Scientific Note</h2>


<div class="scientific-note">


<p>

This system performs
<strong>
image-based screening of suspected microplastic particles
</strong>
from microscopic images.

</p>


<p>

Visual detection alone cannot definitively confirm
that a detected particle is plastic.

</p>


<p>

For laboratory-level confirmation,
techniques such as
<strong>FTIR</strong> or
<strong>Raman spectroscopy</strong>
can be used for polymer identification.

</p>


<p>

<strong>Relative size</strong> is estimated from the
image bounding box.

Actual particle size in micrometers (µm) requires
microscope calibration or a scale reference.

</p>


</div>

</div>


<!-- ================================================= -->
<!-- BACK BUTTON -->
<!-- ================================================= -->

<div class="center">

<a
    href="/"
    class="button"
>
    🔄 Analyze Another Image
</a>

</div>


</div>

</body>

</html>

    """,
        original_url=original_url,
        result_url=result_url,
        detections=detections,
        counts=counts,
        total_particles=total_particles,
        average_confidence=average_confidence,
        detection_level=detection_level,
        possible_sources=possible_sources,
        recommendations=recommendations
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )