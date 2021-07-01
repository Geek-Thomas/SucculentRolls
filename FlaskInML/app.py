from flask import Flask, render_template, jsonify, request
import carts
import id3
import pandas as pd

app = Flask(__name__)

df = pd.read_csv('example_data.csv')
features = df.columns.tolist()
# print(df)

@app.route('/')
def hello_world():
    return render_template('index.html', features=features)

@app.route('/api/get_data')
def get_data():
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
