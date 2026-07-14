"""
==========================================================
 ObjectScanner
 Single File Flask + YOLO Application
 Everything (HTML/CSS/JS) is embedded in this file.
==========================================================
"""

import os
import io
import base64
from datetime import datetime

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    Response
)

from PIL import Image, ImageDraw

from ultralytics import YOLO


############################################################
# Flask Configuration
############################################################

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

############################################################
# Load YOLO Model
############################################################

print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO("best (2).pt")

print("Model loaded successfully.")

############################################################
# App Theme
############################################################

PRIMARY = "#2563eb"
SECONDARY = "#111827"
SUCCESS = "#10b981"
DANGER = "#ef4444"
BACKGROUND = "#0f172a"

############################################################
# Global HTML Header
############################################################

HTML_HEADER = f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1">

<title>Object Scanner</title>

<link
href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap"
rel="stylesheet">

<style>

*{{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Poppins',sans-serif;
}}

body{{
background:{BACKGROUND};
color:white;
}}

nav{{
background:{SECONDARY};
padding:20px;
display:flex;
justify-content:space-between;
align-items:center;
}}

.logo{{
font-size:28px;
font-weight:700;
}}

.container{{
width:90%;
max-width:1100px;
margin:auto;
}}

.hero{{
padding:70px 20px;
text-align:center;
}}

.hero h1{{
font-size:52px;
margin-bottom:20px;
}}

.hero p{{
font-size:20px;
opacity:.8;
margin-bottom:40px;
}}

button{{
background:{PRIMARY};
border:none;
padding:16px 34px;
border-radius:10px;
color:white;
font-size:18px;
cursor:pointer;
transition:.3s;
}}

button:hover{{
transform:scale(1.05);
}}

.card{{
background:#1e293b;
padding:30px;
border-radius:18px;
margin-top:30px;
box-shadow:0 15px 35px rgba(0,0,0,.35);
}}

input[type=file]{{
padding:15px;
width:100%;
}}

img{{
max-width:100%;
border-radius:15px;
margin-top:20px;
}}

.footer{{
padding:30px;
text-align:center;
opacity:.6;
}}

</style>

</head>

<body>

<nav>

<div class="logo">

🧠 ObjectScanner

</div>

<div>

Chair & Monitor Detection

</div>

</nav>

"""
############################################################
# HOME PAGE
############################################################

HOME_PAGE = HTML_HEADER + """

<div class="container">

<div class="hero">

<h1>AI Object Scanner</h1>

<p>

Upload an image or capture one using your device camera.

Our AI will detect Chairs and Monitors instantly.

</p>

<div class="card">

<form
method="POST"
action="/detect"
enctype="multipart/form-data">

<h2 style="margin-bottom:20px;">

Upload Image

</h2>

<input
id="fileInput"
type="file"
name="image"
accept="image/*"
capture="environment"
required>

<br><br>

<img
id="preview"
style="display:none;max-height:500px;">

<br><br>

<button type="submit">

🔍 Scan Image

</button>

</form>

</div>

<br>

<div class="card">

<h2>

How it Works

</h2>

<br>

<div style="font-size:18px;line-height:2;">

📷 Take a picture

<br>

⬆ Upload the image

<br>

🤖 YOLO scans the image

<br>

🪑 If a Chair is detected

you'll be redirected to the Chair page.

<br>

🖥 If a Monitor is detected

you'll be redirected to the Monitor page.

</div>

</div>

</div>

</div>

<div class="footer">

ObjectScanner © 2026

</div>

<script>

const input=document.getElementById("fileInput");

const preview=document.getElementById("preview");

input.onchange=function(e){

const file=e.target.files[0];

if(!file)return;

preview.src=URL.createObjectURL(file);

preview.style.display="block";

}

</script>

</body>

</html>

"""

############################################################
# ROUTES
############################################################

@app.route("/")
def home():

    return render_template_string(HOME_PAGE)
############################################################
# DETECTION ROUTE
############################################################

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:

        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":

        return redirect(url_for("home"))

    ####################################################
    # Read image
    ####################################################

    image = Image.open(file.stream).convert("RGB")

    ####################################################
    # Run YOLO
    ####################################################

    results = model.predict(

        image,

        conf=0.30,

        verbose=False

    )[0]

    ####################################################
    # Draw detections
    ####################################################

    draw = ImageDraw.Draw(image)

    detected_labels = []

    confidences = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        label = str(model.names[cls]).lower()

        detected_labels.append(label)

        confidences.append(conf)

        draw.rectangle(

            [x1, y1, x2, y2],

            outline="red",

            width=4

        )

        draw.text(

            (x1, max(0, y1-20)),

            f"{label} {conf:.2f}",

            fill="red"

        )

    ####################################################
    # Convert image to Base64
    ####################################################

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG")

    encoded = base64.b64encode(

        buffer.getvalue()

    ).decode()

    ####################################################
    # Store globally
    ####################################################

    app.config["LAST_IMAGE"] = encoded

    app.config["LAST_LABELS"] = detected_labels

    app.config["LAST_CONF"] = confidences

    ####################################################
    # Redirect
    ####################################################

    if "chair" in detected_labels:

        return redirect(url_for("chair_page"))

    if (

        "monitor" in detected_labels

        or

        "tv" in detected_labels

        or

        "tvmonitor" in detected_labels

    ):

        return redirect(url_for("monitor_page"))

    return redirect(url_for("not_found"))
############################################################
# CHAIR PAGE
############################################################

@app.route("/chair")
def chair_page():

    image = app.config.get("LAST_IMAGE", "")

    labels = app.config.get("LAST_LABELS", [])

    confs = app.config.get("LAST_CONF", [])

    highest = 0

    for label, conf in zip(labels, confs):

        if label == "chair":

            highest = max(highest, conf)

    PAGE = HTML_HEADER + f"""

<div class="container">

<div class="hero">

<h1>

🪑 Chair Detected

</h1>

<p>

The AI model successfully detected one or more chairs.

</p>

</div>

<div class="card">

<h2>

Detection Result

</h2>

<br>

<img
src="data:image/jpeg;base64,{image}">

<br><br>

<h2>

Confidence

</h2>

<h1
style="color:#10b981;">

{highest*100:.2f}%

</h1>

</div>

<div class="card">

<h2>

About Chairs

</h2>

<br>

<p
style="font-size:18px;
line-height:2;">

A chair is one of the most common pieces of furniture
used to support a seated person.

Chairs are manufactured using

wood,

plastic,

metal,

steel,

aluminium,

fabric,

mesh,

and composite materials.

Modern ergonomic chairs are specifically designed to
reduce strain on the back, neck and shoulders.

</p>

</div>

<div class="card">

<h2>

Common Uses

</h2>

<br>

<ul
style="font-size:18px;
line-height:2;
text-align:left;">

<li>Office Workstations</li>

<li>Schools & Universities</li>

<li>Dining Rooms</li>

<li>Hospitals</li>

<li>Airports</li>

<li>Restaurants</li>

<li>Waiting Lounges</li>

</ul>

</div>

<div class="card">

<h2>

Interesting Facts

</h2>

<br>

<p
style="font-size:18px;
line-height:2;">

✔ Ergonomic chairs reduce fatigue.

<br>

✔ Gaming chairs provide lumbar support.

<br>

✔ Office chairs usually support
adjustable height and reclining.

<br>

✔ Mesh chairs improve airflow.

</p>

</div>

<div
style="text-align:center;
margin:60px;">

<a href="/">

<button>

🔍 Scan Another Image

</button>

</a>

</div>

</div>

<div class="footer">

Chair Detection Complete

</div>

</body>

</html>

"""

    return render_template_string(PAGE)
############################################################
# MONITOR PAGE
############################################################

@app.route("/monitor")
def monitor_page():

    image = app.config.get("LAST_IMAGE", "")

    labels = app.config.get("LAST_LABELS", [])

    confs = app.config.get("LAST_CONF", [])

    highest = 0

    for label, conf in zip(labels, confs):

        if label in ["monitor", "tv", "tvmonitor"]:

            highest = max(highest, conf)

    PAGE = HTML_HEADER + f"""

<div class="container">

<div class="hero">

<h1>

🖥️ Monitor Detected

</h1>

<p>

The AI model successfully detected a computer monitor.

</p>

</div>

<div class="card">

<h2>

Detection Result

</h2>

<br>

<img
src="data:image/jpeg;base64,{image}">

<br><br>

<h2>

Confidence Score

</h2>

<h1 style="color:#10b981;">

{highest*100:.2f}%

</h1>

</div>

<div class="card">

<h2>

About Computer Monitors

</h2>

<br>

<p
style="
font-size:18px;
line-height:2;
">

A monitor is an electronic display used
to present visual information from a computer.

Modern monitors are available in

LCD,

LED,

OLED,

Mini-LED

and curved display technologies.

Monitors are essential in

software development,

gaming,

education,

medical imaging,

engineering

and professional content creation.

</p>

</div>

<div class="card">

<h2>

Common Applications

</h2>

<br>

<ul
style="
font-size:18px;
line-height:2;
text-align:left;
">

<li>Programming</li>

<li>Gaming</li>

<li>Video Editing</li>

<li>Graphic Design</li>

<li>Medical Diagnostics</li>

<li>Office Productivity</li>

<li>CAD & Engineering</li>

</ul>

</div>

<div class="card">

<h2>

Interesting Facts

</h2>

<br>

<p
style="
font-size:18px;
line-height:2;
">

✔ OLED monitors have individually lit pixels.

<br>

✔ High refresh rates improve gaming.

<br>

✔ IPS panels provide better colour accuracy.

<br>

✔ 4K monitors contain over 8 million pixels.

</p>

</div>

<div
style="
margin:60px;
text-align:center;
">

<a href="/">

<button>

🔍 Scan Another Image

</button>

</a>

</div>

</div>

<div class="footer">

Monitor Detection Complete

</div>

</body>

</html>

"""

    return render_template_string(PAGE)
############################################################
# NO OBJECT DETECTED PAGE
############################################################

@app.route("/notfound")
def not_found():

    PAGE = HTML_HEADER + """

<div class="container">

<div class="hero">

<h1>

❌ No Supported Object Detected

</h1>

<p>

The AI could not find a Chair or a Monitor in the uploaded image.

</p>

</div>

<div class="card">

<h2>

Possible Reasons

</h2>

<br>

<ul
style="
font-size:18px;
line-height:2;
text-align:left;
">

<li>The image is blurry.</li>

<li>The object is partially hidden.</li>

<li>The object is too small.</li>

<li>The confidence score was below the detection threshold.</li>

<li>The uploaded object is not a Chair or Monitor.</li>

</ul>

</div>

<div class="card">

<h2>

Tips for Better Detection

</h2>

<br>

<ul
style="
font-size:18px;
line-height:2;
text-align:left;
">

<li>📷 Capture the entire object.</li>

<li>💡 Ensure good lighting.</li>

<li>📏 Move closer to the object.</li>

<li>🖼 Use a high-resolution image.</li>

<li>🚫 Avoid excessive blur or glare.</li>

</ul>

</div>

<div
style="
text-align:center;
margin:60px;
">

<a href="/">

<button>

🔄 Try Another Image

</button>

</a>

</div>

</div>

<div class="footer">

ObjectScanner • Detection Finished

</div>

</body>

</html>

"""

    return render_template_string(PAGE)
############################################################
# MAIN
############################################################
 

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 ObjectScanner Starting...")
    print("=" * 60)
    print("Model Loaded Successfully")
    print("Open your browser at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)


    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
