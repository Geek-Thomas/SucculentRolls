import os

from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import carts
import id3
import pandas as pd
import settings

app = Flask(__name__)
app.secret_key = 'ajifjoefjiwn1'
app.config.from_object(settings)
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# print(df)
df = pd.DataFrame()
features = df.columns.tolist()
upload = False

# 文件过滤器
def allowed_files(filename):
    return '.' in filename and filename.rsplit('.')[1].lower() in ALLOWED_EXTENSIONS

# 读取上传的文件
def read_file(filename):
    if 'xlsx' in filename or 'xls' in filename:
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename)
    features = df.columns.tolist()
    return df, features

@app.route('/')
def index():
    return render_template('index.html', features=features)

# 上传文件
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global upload
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_files(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(settings.UPLOAD_FOLDER, filename))
            upload = True
            flash("文件上传成功")
            return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/api/get_data')
def get_data():
    global df, features
    if upload:
        filename = os.listdir(settings.UPLOAD_FOLDER)[-1]
        df, features = read_file(filename)
    return {
        'tableData': df.to_dict(orient='records')
    }

@app.route('/api/types/')
def cart():
    algorithm = request.args.get('algorithm')
    if algorithm == 'cart':
        dataset, features = carts.createDataset()
        myTree = carts.createTree(dataset, features)
        carts.createPlot(myTree)
        return {'myTree': myTree}
    elif algorithm == 'id3':
        dataset, features = id3.createDataset()
        myTree = id3.createDecisionTree(dataset, features)
        id3.createPlot(myTree)
        return {'myTree': myTree}
    else:
        return ''


if __name__ == '__main__':
    app.run()
