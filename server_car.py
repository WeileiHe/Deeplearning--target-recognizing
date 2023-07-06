from flask import Flask, render_template, request, redirect, url_for
import subprocess
import pandas as pd
import os
import shutil

app = Flask(__name__,static_folder = 'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display/<image_id>/<result>')
def display(image_id, result):
    return render_template('display_car.html', image_id=image_id, result=result)

@app.route('/detect', methods=['POST'])
def detect():
    image_id = request.form.get('image_id')

    # Copy the image to the Samples folder
    shutil.copy(os.path.join('data/images', image_id + '.jpg'), 'data/Samples')

    # Run the detect.py script
    exp_num = max([int(dir_name[3:]) for dir_name in os.listdir('output') if dir_name.startswith('exp')], default=0) + 1
    command = f"python3 detect.py --source 'data/Samples' --weights 'runs/train/exp25/weights/best.pt' --device cpu --conf 0.5 --iou 0.45 --project 'output' --name exp{exp_num} --save-txt"
    process = subprocess.Popen(command, shell=True)
    process.wait()

    # Read and sort the results
    txt_file = f"output/exp{exp_num}/labels/{image_id}.txt"
    df = pd.read_csv(txt_file, delimiter = " ", header=None)
    df.sort_values(by=[1], inplace=True)
    
    # Map first column numbers to characters
    classes = ["", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
               "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
               "Y", "Z", "澳", "川", "鄂", "甘", "赣", "港", "贵", "桂",
               "黑", "沪", "吉", "冀", "津", "晋", "京", "警", "辽", "鲁",
               "蒙", "闽", "宁", "青", "琼", "陕", "苏", "皖", "湘", "新",
               "学", "渝", "豫", "粤", "云", "浙", "藏"]
    df[0] = df[0].map(lambda x: classes[x])
    
    # Convert results to a string
    result = ''.join(df[0].tolist())

    return redirect(url_for('display', image_id=image_id, result=result))

if __name__ == '__main__':
    app.run(debug=True)
