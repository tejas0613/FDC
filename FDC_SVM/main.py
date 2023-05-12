from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
import os
import base64
import io
from io import BytesIO
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')

app = Flask(__name__)  
model = pickle.load(open('svm_poly.sav', 'rb'))

def get_plot(data, sel_features):
    # x = np.arange(len(data))
    # fig = plt.figure()
    # ax = fig.subplots()
    # ax.plot(values, c='darkblue', marker='o')
    data_size = 45
    x_sel, y_sel = sel_features
    normal_df = data[data['ND'] == 1]
    X_normal = normal_df['UC']
    Y_normal = normal_df['LBE']
    X_normal = X_normal.tolist()[:data_size]
    Y_normal = Y_normal.tolist()[:data_size]
    plt.scatter(X_normal, Y_normal, color = '#43A367')
    distress_df = data[data['ND'] == 2]
    X_dist = distress_df['UC']
    Y_dist = distress_df['LBE']
    X_dist = X_dist.tolist()[:data_size]
    Y_dist = Y_dist.tolist()[:data_size]
    plt.scatter(X_dist, Y_dist, color = 'hotpink')
    plt.scatter(x_sel, y_sel, color = 'yellow', s = 80)
    plt.legend(['Normal', 'Distress','Selected Patient'], bbox_to_anchor = (0.75, 1.15), ncol = 2)
    plt.xlabel('Uterine Contractions (UC)')
    plt.ylabel('Baseline Fetal Heart Rate (LBE)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plot = base64.b64encode(buf.getbuffer()).decode('ascii')
    plt.close()
    return plot

# def get_plot(data):
#     plt.scatter(data,np.arange(len(data)), c='r', marker='o')
#     return plt

#Cleaning the data file for uploading
# df = pd.read_csv('CTG - New data.csv')
# df.drop(['FileName','Date','b','e','ASTV','MSTV','ALTV','ALTV','MLTV','Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode',
#        'Mean', 'Median', 'Variance', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD',
#        'DE', 'LD', 'FS', 'SUSP', 'CLASS'], axis=1, inplace=True)
# print(df.head())
# x = 1561
# df['SegFile'] = df['SegFile'].apply(lambda f: f.split('.')[0]) #['CTG1562', 'txt']
# df['SegFile'] = df['SegFile'].apply(lambda f: f.split('G')[1]) #['CT', '1562']
# df['SegFile'].astype(int)
# df['SegFile'] = df['SegFile'].apply(lambda f: int(f) - int(x))
# print(df.head())
# df.to_csv('cleandata.csv', index = None)

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/model', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()] #[123,123,0,1,....]
    Scalar = StandardScaler()
    standard_features = Scalar.fit_transform(np.array(features).reshape(1, -1))
    array = standard_features
    final_features = np.array(standard_features).reshape(1,-1)
    result = model.predict(final_features)
    output = round(result[0], 2)
    if output == 1.0:
        result = 'Normal'
    elif output == 2.0:
        result = 'Distressed'
    plot = get_plot(features)
    # plot.savefig(os.path.join('static', 'plot.png'))
    # plot.close()
    return render_template('analysis.html', **locals())

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        file.save('uploadedfile.csv')
        data = pd.read_csv('uploadedfile.csv')
        index = int(request.form['index'])
        idx = np.where(data['SegFile'] == index)[0][0] 
        temp = data.to_numpy()
        sel_features = (temp[idx][5], temp[idx][1])
        plot = get_plot(data, sel_features)
        data.drop(['SegFile', 'ND'], axis=1, inplace=True)
        features = data.to_numpy()
        Scalar = StandardScaler()
        standard_features = Scalar.fit_transform(data)
        #standard_features = standard_features.to_numpy()
        standard_features = standard_features.astype(float)
        array = features[idx] #[[122,122,0,1,5,1,0,0,0]] 
        ffarr = array.reshape(1,-1)
        result = model.predict(standard_features[idx].reshape(1,-1))
        output = round(result[0], 2)
        if output == 1.0:
            result = 'Normal'
        elif output == 2.0:
            result = 'Distressed'
        
        # plot.savefig(os.path.join('static', 'plot.png'))
        # plot.close()
        return render_template('analysis.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)

# data = pd.read_csv('uploadedfile.csv')
# index = 6
# idx = np.where(data['SegFile'] == index)[0][0]
# data.drop(['SegFile', 'ND'], axis=1, inplace=True)
# features = data.to_numpy()
# print(f"idx = {idx}")
# print(features[idx])