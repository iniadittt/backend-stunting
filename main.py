import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from flask import Flask, jsonify, request
from flask_mysqldb import MySQL
from dotenv import load_dotenv
from http import HTTPStatus
from tensorflow.keras.models import load_model
from datetime import date
from urllib.parse import urlparse

load_dotenv()

parsed_url = urlparse(os.getenv('DATABASE_URL'))
print(parsed_url)
host = parsed_url.hostname
user = parsed_url.username
password = parsed_url.password
database = parsed_url.path.lstrip('/')

app = Flask(__name__)
app.config['MYSQL_HOST'] = host
app.config['MYSQL_USER'] = user
app.config['MYSQL_PASSWORD'] = password
app.config['MYSQL_DB'] = database
mysql = MySQL(app)

app.config['MODEL'] = './model/stunting.h5'
model = load_model(app.config['MODEL'], compile=False)

@app.route('/')
def hello_world():
    return jsonify({
        'code': HTTPStatus.OK,
        'message': 'Hello, World!'
    })

@app.route('/predict', methods=['POST'])
def prediksi():
    if request.method == 'POST':
        try:
            data = request.get_json()
            id_anak = int(data['id_anak'])
            tb = float(data['tb'])
            bb = float(data['bb'])
            usia = int(data['usia'])
            if not id_anak or not tb or not bb or not usia:
                raise KeyError("Data tidak sesuai, data dibutuhkan: 'nama', 'nama_ibu', 'jk'")
            if not isinstance(id_anak, int) or not isinstance(usia, int):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'id_anak' or 'usia'. Data harus integer."
                }), HTTPStatus.BAD_REQUEST
            if not isinstance(tb, float) or not isinstance(bb, float):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'tb' or 'bb'. Data harus float."
                }), HTTPStatus.BAD_REQUEST
            cur = mysql.connection.cursor()
            cur.execute('''SELECT * FROM anak WHERE id = %s''', (id_anak,))
            data_anak = cur.fetchone()
            cur.close()
            if not data_anak:
                return jsonify({
                    'code': HTTPStatus.NOT_FOUND,
                    'message': 'Data anak tidak ada',
                }), HTTPStatus.NOT_FOUND
            jk = 0 if data_anak[3] == 'Laki-laki' else 1
            tbm2 = (tb / 100) ** 2
            imt = bb / tbm2
            features = [tb, bb, imt, usia, int(jk)]
            np_data = np.array(features).reshape(1, -1)
            response = model.predict(np_data)
            max_index = np.argmax(response)
            conditions = ['Stunting', 'Normal', 'Obesitas']
            condition = conditions[max_index] if max_index < len(conditions) else 'Not found'
            percentage = response[0][max_index] * 100 
            return jsonify({
                'code': HTTPStatus.OK,
                'message': 'Berhasil predicting',
                'data': {
                    'imt': imt,
                    'persentase': percentage,
                    'kondisi': condition
                }
            }), HTTPStatus.OK
        except KeyError as e:
            return jsonify({
                'code': HTTPStatus.BAD_REQUEST,
                'message': f'Missing key: {e}'
            }), HTTPStatus.BAD_REQUEST
        except Exception as e:
            return jsonify({
                'code': HTTPStatus.INTERNAL_SERVER_ERROR,
                'message': f'An error occurred: {e}'
            }), HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        return jsonify({
            'code': HTTPStatus.METHOD_NOT_ALLOWED,
            'message': 'Method not allowed'
        }), HTTPStatus.METHOD_NOT_ALLOWED
        
@app.route('/anak')
def get_anak():
    try:
        cur = mysql.connection.cursor()
        cur.execute('''SELECT * FROM anak''')
        data = cur.fetchall()
        print(data)
        cur.close()
        if not data:
            return jsonify({
                'code': HTTPStatus.NOT_FOUND,
                'message': 'Data anak tidak ada'
            }), HTTPStatus.NOT_FOUND
        anak_list = []
        for row in data:
            anak = {
                'id': row[0],
                'nama': row[1],
                'nama_ibu': row[2],
                'jk': row[3]
            }
            anak_list.append(anak)
        return jsonify({
            'code': HTTPStatus.OK,
            'message': 'Berhasil mengambil data anak',
            'data': {
                'anak': anak_list
            }
        }), HTTPStatus.OK
    except Exception as e:
        return jsonify({
            'code': HTTPStatus.INTERNAL_SERVER_ERROR,
            'message': f'An error occurred: {e}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR
        
@app.route('/anak', methods=['POST'])
def post_anak():
    if request.method == 'POST':
        try:
            data = request.get_json()
            nama = data['nama']
            nama_ibu = data['nama_ibu']
            jk = data['jk']
            if not nama or not nama_ibu or not jk:
                raise KeyError("Data tidak sesuai, data dibutuhkan: 'nama', 'nama_ibu', 'jk'")
            if not isinstance(nama, str) or not isinstance(nama_ibu, str) or not isinstance(jk, str):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'nama' or 'nama_ibu' or 'jk'. Data harus string."
                }), HTTPStatus.BAD_REQUEST
            if jk not in ['Laki-laki', 'Perempuan']:
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'jk'. Data harus 'Laki-laki' or 'Perempuan'."
                }), HTTPStatus.BAD_REQUEST
            cur = mysql.connection.cursor()
            cur.execute('''INSERT INTO anak (nama, nama_ibu, jk) VALUES (%s, %s, %s)''', (nama, nama_ibu, jk))
            mysql.connection.commit()
            cur.close()
            return jsonify({
                'code': HTTPStatus.CREATED,
                'message': 'Data anak berhasil ditambahkan'
            }), HTTPStatus.CREATED
        except KeyError as e:
            return jsonify({
                'code': HTTPStatus.INTERNAL_SERVER_ERROR,
                'message': f'An error occurred: {e}'
            }), HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        return jsonify({
            'code': HTTPStatus.METHOD_NOT_ALLOWED,
            'message': 'Method not allowed'
        }), HTTPStatus.METHOD_NOT_ALLOWED

@app.route('/history')
def get_history():
    try:
        cur = mysql.connection.cursor()
        cur.execute('''
            SELECT 
                h.id AS history_id,
                a.id AS anak_id,
                a.nama AS anak_nama,
                a.nama_ibu AS anak_nama_ibu,
                a.jk AS anak_jk,
                h.tanggal,
                h.tb,
                h.bb,
                h.imt,
                h.usia,
                h.kondisi
            FROM 
                history h
            JOIN 
                anak a ON h.anak_id = a.id;
        ''')
        data = cur.fetchall()
        cur.close()
        if not data:
            return jsonify({
                'code': HTTPStatus.NOT_FOUND,
                'message': 'Data history tidak ada'
            }), HTTPStatus.NOT_FOUND
        history_list = []
        for row in data:
            history = {
                'id': row[0],
                'anak': {
                    'id': row[1],
                    'nama': row[2],
                    'nama_ibu': row[3],
                    'jk': row[4],
                },
                'tanggal': row[5],
                'tb': row[6],
                'bb': row[7],
                'imt': row[8],
                'usia': row[9],
                'kondisi': row[10],
            }
            history_list.append(history)
        return jsonify({
            'code': HTTPStatus.OK,
            'message': 'Berhasil mengambil data history',
            'data': {
                'history': history_list
            }
        }), HTTPStatus.OK
    except Exception as e:
        return jsonify({
            'code': HTTPStatus.INTERNAL_SERVER_ERROR,
            'message': f'An error occurred: {e}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR
        
@app.route('/history/<int:id>')
def get_history_anak(id):
    try:
        cur = mysql.connection.cursor()
        cur.execute('''
            SELECT 
                h.id AS history_id,
                a.id AS anak_id,
                a.nama AS anak_nama,
                a.nama_ibu AS anak_nama_ibu,
                a.jk AS anak_jk,
                h.tanggal,
                h.tb,
                h.bb,
                h.imt,
                h.usia,
                h.kondisi
            FROM 
                history h
            JOIN 
                anak a ON h.anak_id = a.id
            WHERE 
                h.anak_id = %s
        ''', (id,))
        data = cur.fetchall()
        cur.close()
        if not data:
            return jsonify({
                'code': HTTPStatus.NOT_FOUND,
                'message': 'Data history tidak ada'
            }), HTTPStatus.NOT_FOUND
        history_list = []
        for row in data:
            history = {
                'id': row[0],
                'anak': {
                    'id': id,
                    'nama': row[2],
                    'nama_ibu': row[3],
                    'jk': row[4],
                },
                'tanggal': row[5],
                'tb': row[6],
                'bb': row[7],
                'imt': row[8],
                'usia': row[9],
                'kondisi': row[10],
            }
            history_list.append(history)
        return jsonify({
            'code': HTTPStatus.OK,
            'message': 'Berhasil mengambil data history',
            'data': {
                'history': history_list
            }
        }), HTTPStatus.OK
    except Exception as e:
        return jsonify({
            'code': HTTPStatus.INTERNAL_SERVER_ERROR,
            'message': f'An error occurred: {e}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR
        
@app.route('/history', methods=['POST'])
def post_history():
    if request.method == 'POST':
        try:
            data = request.get_json()
            id_anak = data['id_anak']
            tanggal = data['tanggal']
            tb = float(data['tb'])
            bb = float(data['bb'])
            usia = data['usia']
            kondisi = data['kondisi']
            imt = tbm2 = (tb / 100) ** 2
            imt = float(bb / tbm2)
            if not id_anak or not tanggal or not tb or not bb or not usia or not kondisi:
                raise KeyError("Data tidak sesuai, data dibutuhkan: 'id_anak', 'tanggal', 'tb', 'bb', 'usia', 'kondisi'")
            if not isinstance(id_anak, int) or not isinstance(usia, int):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'id_anak' or 'usia'. Data harus integer."
                }), HTTPStatus.BAD_REQUEST
            if not isinstance(tanggal, str) or not isinstance(kondisi, str):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'tanggal' or 'kondisi'. Data harus string."
                }), HTTPStatus.BAD_REQUEST
            if not isinstance(tb, float) or not isinstance(bb, float):
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'tb' or 'bb'. Data harus float."
                }), HTTPStatus.BAD_REQUEST
            if kondisi not in ['Stunting', 'Normal', 'Obesitas']:
                return jsonify({
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': "Value tidak sesuai untuk 'kondisi'. Data harus 'Stunting' or 'Normal' or 'Obesitas'."
                }), HTTPStatus.BAD_REQUEST
            cur = mysql.connection.cursor()
            cur.execute('''SELECT * FROM anak WHERE id = %s''', (id_anak,))
            data_anak = cur.fetchone()
            cur.close()
            if not data_anak:
                return jsonify({
                    'code': HTTPStatus.NOT_FOUND,
                    'message': 'Data anak tidak ditemukan'
                }), HTTPStatus.NOT_FOUND
            cur = mysql.connection.cursor()
            cur.execute('''
                INSERT INTO history (anak_id, tanggal, tb, bb, imt, usia, kondisi)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (id_anak, tanggal, tb, bb, imt, usia, kondisi))
            mysql.connection.commit()
            cur.close()
            return jsonify({
                'code': HTTPStatus.CREATED,
                'message': 'Data history berhasil ditambahkan'
            }), HTTPStatus.CREATED
        except KeyError as e:
            return jsonify({
                'code': HTTPStatus.INTERNAL_SERVER_ERROR,
                'message': f'An error occurred: {e}'
            }), HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        return jsonify({
            'code': HTTPStatus.METHOD_NOT_ALLOWED,
            'message': 'Method not allowed'
        }), HTTPStatus.METHOD_NOT_ALLOWED
        
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))