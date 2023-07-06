from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN

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
    image_id_len = len(image_id)

    if image_id_len == 6:
        # Follow the first script logic
        src = f"data/images/{image_id}.jpg"
        dest = "data/Samples"
        shutil.copy(src, dest)
        exp_num = max([int(dir_name[3:]) for dir_name in os.listdir('output') if dir_name.startswith('exp')], default=0) + 1
        command = f"python3 detect.py --source 'data/Samples' --weights 'runs/train/exp17/weights/best.pt' --device cpu --conf 0.6 --iou 0.45 --project 'output' --name exp{exp_num} --save-txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()
        txt_file = f"output/exp{exp_num}/labels/{image_id}.txt"
        df = pd.read_csv(txt_file, delimiter = " ", header=None)
        df.sort_values(by=[1], inplace=True)
        result = [number_to_char[i] for i in df[0].tolist()]
        return redirect(url_for('display', image_id=image_id, result=result))

    elif image_id_len == 4:
        # Follow the second script logic
        shutil.copy(os.path.join('data/images', image_id + '.jpg'), 'data/Samples')
        exp_num = max([int(dir_name[3:]) for dir_name in os.listdir('output') if dir_name.startswith('exp')], default=0) + 1
        command = f"python3 detect.py --source 'data/Samples' --weights 'runs/train/exp25/weights/best.pt' --device cpu --conf 0.5 --iou 0.45 --project 'output' --name exp{exp_num} --save-txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()
        txt_file = f"output/exp{exp_num}/labels/{image_id}.txt"
        df = pd.read_csv(txt_file, delimiter = " ", header=None)
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(df[[2]])
        df['Cluster'] = -clustering.labels_

    # Sorting by cluster label and then by x-coordinate
        df.sort_values(by=['Cluster', 1], ascending=[True, True], inplace=True)
        classes = ["", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "澳", "川", "鄂", "甘", "赣", "港", "贵", "桂", "黑", "沪", "吉", "冀", "津", "晋", "京", "警", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕", "苏", "皖", "湘", "新", "学", "渝", "豫", "粤", "云", "浙", "藏"]
        df[0] = df[0].map(lambda x: classes[x])
        current_cluster = df.iloc[0]['Cluster']
        result = ''
        for _, row in df.iterrows():
            if row['Cluster'] != current_cluster:
                result += '<br>'
                current_cluster = row['Cluster']
            result += row[0]
        return redirect(url_for('display_car', image_id=image_id, result=result))
    elif image_id_len == 8 :
        shutil.copy(os.path.join('data/images', image_id + '.jpg'), 'data/Samples')
        exp_num = max([int(dir_name[3:]) for dir_name in os.listdir('output') if dir_name.startswith('exp')], default=0) + 1
        command = f"python3 detect.py --source 'data/Samples' --weights 'runs/train/exp26/weights/best.pt' --device cpu --conf 0.44 --iou 0.45 --project 'output' --name exp{exp_num} --save-txt"
        process = subprocess.Popen(command, shell=True)
        process.wait()
        txt_file = f"output/exp{exp_num}/labels/{image_id}.txt"
        df = pd.read_csv(txt_file, delimiter = " ", header=None)
        
# Your given class dictionary
        classes = ["numberarea", "specialarea", "ding", "carcode", "", "", "", "", "type_heng", "type_shu",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_0", "_1", "_2", "_3", "_4", "_5", "_6",
        "_7", "_8", "_9", "closed", "opened"]

        df[0] = df[0].map(lambda x: classes[x])
        result = ""
        if 'type_heng' in df[0].values:  # Horizontal
    # Remove 'type_heng'
            df = df[df[0] != 'type_heng']
    
    # Cluster by y values
            dbscan = DBSCAN(eps=0.02, min_samples=1)  # Adjust parameters as necessary
            df['row'] = dbscan.fit_predict(df[2].values.reshape(-1, 1))

    # Calculate mean y value for each row
            mean_y = df.groupby('row')[2].mean().sort_values(ascending=True)
            sorted_rows = mean_y.index.tolist()

    # Sort by row (from top to bottom) then x values (from left to right)
            df['row'] = df['row'].map(lambda x: sorted_rows.index(x))
            df.sort_values(['row', 1], ascending=[True, True], inplace=True)
            last_row = df['row'].iloc[0]
            for _, row in df.iterrows():
                if row['row'] != last_row:
                    result += '\n'
                result += row[0]
                last_row = row['row']
            result = result.replace("\n", "<br>")
        elif 'type_shu' in df[0].values:  # Vertical
    # Remove 'type_shu'
            df = df[df[0] != 'type_shu']

    # Cluster by x values
            dbscan = DBSCAN(eps=0.03, min_samples=1)  # Adjust parameters as necessary
            df['col'] = dbscan.fit_predict(df[1].values.reshape(-1, 1))

    # Calculate mean x value for each col
            mean_x = df.groupby('col')[1].mean().sort_values(ascending=True)
            sorted_cols = mean_x.index.tolist()

    # Sort by col (from left to right) then y values (from top to bottom)
            df['col'] = df['col'].map(lambda x: sorted_cols.index(x))
            df.sort_values(['col', 2], ascending=[True, True], inplace=True)
            last_col = df['col'].iloc[0]
            for _, row in df.iterrows():
                if row['col'] != last_col:
                    result += '\n'
                result += row[0]
                last_col = row['col']
            result = result.replace("\n", "<br>")
        return redirect(url_for('display_casenumber', image_id=image_id, result=result))
    else:
        # Invalid image_id
        return "Invalid image_id!", 400

@app.route('/display')
def display():
    image_id = request.args.get('image_id')
    result = request.args.getlist('result')
    return render_template('display.html', image_id=image_id, result=result)
@app.route('/display_car/<image_id>/<result>')

def display_car(image_id, result):
    return render_template('display_car.html', image_id=image_id, result=result)

@app.route('/display_casenumber/<image_id>/<result>')
def display_casenumber(image_id, result):
    return render_template('display_casenumber.html', image_id = image_id, result = result)

if __name__ == '__main__':
    app.run(debug=True)

