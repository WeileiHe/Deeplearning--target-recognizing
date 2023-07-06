from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
import shutil
import pandas as pd

app = Flask(__name__, static_folder='data')
number_to_char = {i: chr(65 + i) for i in range(26)}  # 0-25 -> 'A'-'Z'
number_to_char.update({i: chr(71 + i) for i in range(26, 52)})  # 26-51 -> 'a'-'z'
number_to_char.update({i: chr(i - 4) for i in range(52, 62)})  # 52-61 -> '0'-'9'
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image_id = request.form.get('image_id')

    # Assume the images are in 'data/Images' folder
    src = f"data/images/{image_id}.jpg"
    dest = "data/Samples"

    # Move the image to 'Samples' folder
    shutil.copy(src, dest)

    # Run the detect.py script
    exp_num = max([int(dir_name[3:]) for dir_name in os.listdir('output') if dir_name.startswith('exp')], default=0) + 1
    command = f"python3 detect.py --source 'data/Samples' --weights 'runs/train/exp17/weights/best.pt' --device cpu --conf 0.6 --iou 0.45 --project 'output' --name exp{exp_num} --save-txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    # Assume the txt files are in 'output' folder
    txt_file = f"output/exp{exp_num}/labels/{image_id}.txt"

    # Read the txt file and sort
    df = pd.read_csv(txt_file, delimiter = " ", header=None)
    df.sort_values(by=[1], inplace=True)

    # Return the first column
    result = [number_to_char[i] for i in df[0].tolist()]

    return redirect(url_for('display', image_id=image_id, result=result))

@app.route('/display')
def display():
    image_id = request.args.get('image_id')
    result = request.args.getlist('result')
    return render_template('display.html', image_id=image_id, result=result)

if __name__ == '__main__':
    app.run(debug=True)
