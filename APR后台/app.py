from flask import Flask, request
from flask_cors import *
import json as json
# from algor2 import vrp
from main_gene import vrp
import os
import time
import json
import numpy as np



app = Flask(__name__)
CORS(app, supports_credentials=True)

id_dis = 1
id_coo = 1

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/solve', methods=['POST'])
def solve():
    print("------------------------------------------")
    print('Test solve  start.')

    # 如果请求体的数据不是表单格式的（如json格式，xml格式），可以通过request.data获取
    # print(request.data)  # {"name": "zhangsan", "age": 18}
    data = json.loads(request.data)
    num = float(data['num'])
    dis = data['dis']
    dem = data['dem']
    carLoad = data['carLoad']
    edge_max = float(data['edge_max'])
    loop_max = float(data['loop_max'])
    speed = float(data['speed'])
    uTime = float(data['uTime'])

    full = float(data['fac_full'])
    distance = float(data['fac_distance'])
    tim = float(data['fac_time'])
    iter = int(data['max_iter'])
    population = int(data['max_population'])

    # print('-----------------------------------------')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    v = vrp(num, dis, dem, carLoad, edge_max, loop_max, speed, uTime)
    ans = v.solve(start_num=150, max_iter=iter, max_population=population, fac_full=full,
                  fac_distance=distance, fac_time=tim, show=True, dia=True)
    ans = json.dumps(ans, cls=MyEncoder)
    return ans


# @app.route('/csv_dis', methods=['POST'])
# def csv_dis():
#     print("------------------------------------------")
#     print('save dis csv:' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#
#
#     f = request.files['file']
#     #data = json.loads(request.data)
#     full = int(request.form.get('fac_full'))
#     distance = int(request.form.get('fac_distance'))
#     tim = int(request.form.get('fac_time'))
#
#     global id_dis
#
#     Dir_name = "WebTransUserData/" + str(id_dis) + '/'
#
#     id_dis = id_dis + 1
#
#     os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹
#
#     f.save(Dir_name + 'csv_dis.csv')
#
#     v = v2_vrp(Dir_name + 'csv_dis.csv', 'dis')
#     ans = v.solve(start_num=150, max_iter=100, max_population=1500, fac_full=full,
#                   fac_distance=distance, fac_time=tim, show=True, dia=True)
#     ans = json.dumps(ans, cls=MyEncoder)
#     return ans
#
#
# @app.route('/csv_coo', methods=['POST'])
# def csv_coo():
#     print("------------------------------------------")
#     print('save coo csv:' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#
#
#     f = request.files['file']
#     full = int(request.form.get('fac_full'))
#     distance = int(request.form.get('fac_distance'))
#     tim = int(request.form.get('fac_time'))
#
#     global id_coo
#
#     Dir_name = "WebTransUserData/" + str(id_coo) + '/'
#
#     id_coo = id_coo + 1
#
#     os.makedirs(Dir_name, exist_ok=True)  # 以该名称建一个文件夹
#
#     f.save(Dir_name + 'csv_coo.csv')
#
#     v = v2_vrp(Dir_name + 'csv_coo.csv', 'coo')
#     ans = v.solve(start_num=150, max_iter=100, max_population=1500, fac_full=full,
#                   fac_distance=distance, fac_time=tim, show=True, dia=True)
#     ans = json.dumps(ans, cls=MyEncoder)
#     return ans


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == '__main__':
    app.run()
