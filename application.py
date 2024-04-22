import pymysql
import sys
import os
import json
import logging
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# 환경 변수에서 데이터베이스 정보 읽기
DATABASE_HOST = os.getenv('DB_HOST', 'localhost')
DATABASE_USER = os.getenv('DB_USER', 'user')
DATABASE_PASSWORD = os.getenv('DB_PASSWORD', 'password')
DATABASE_DB = os.getenv('DB_NAME', 'database')

# 모델은 글로벌 변수로 한 번만 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def get_db_connection():
    connection = pymysql.connect(host=DATABASE_HOST,
                                 user=DATABASE_USER,
                                 password=DATABASE_PASSWORD,
                                 db=DATABASE_DB,
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection

@app.route('/')
def index():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM temp_que_ans")
            result = cursor.fetchall()
        return str(result)
    finally:
        if connection is not None:
            connection.close()

@app.route('/saju2/', methods=['POST'])
def saju2():
    connection = None
    try:
        connection = get_db_connection()
        text = request.form['title']
        f_obj = request.form['f_obj']

        query = "SELECT * FROM temp_que_ans WHERE key_01 = %s"
        cursor = connection.cursor()
        cursor.execute(query, (f_obj,))
        rows = cursor.fetchall()

        max_distance = -1
        best_answer = None
        embedding = model.encode([text])[0]

        for row in rows:
            emb = json.loads(row['emb'])
            distance = cosine_similarity([embedding], [emb])[0][0]
            if distance > max_distance:
                max_distance = distance
                best_answer = row

        if best_answer:
            response = {
                'a': str(best_answer['key_01']),
                'b': best_answer['que'],
                'c': best_answer['ans'],
                'wr_id': int(best_answer['wr_id']),
                'key_02': int(best_answer['key_02']),
                'distance': float(max_distance)
            }
            return jsonify(response)
        else:
            return jsonify({"error": "No matching record found"})
    
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)})
    
    finally:
        if connection is not None:
            connection.close()

@app.route('/osan_csv_kor_emd/', methods=['POST'])
def osan_csv_kor_emd():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT wr_id, key_01, key_02, key_03, que FROM temp_que_ans WHERE CHAR_LENGTH(emb) < 10;")
            rows = cursor.fetchall()

            for row in rows:
                wr_id = row['wr_id']
                que = row['que']
                embeddings = model.encode([que])
                embedding_str = json.dumps(embeddings[0].tolist())

                update_query = "UPDATE temp_que_ans SET emb = %s WHERE wr_id = %s"
                cursor.execute(update_query, (embedding_str, wr_id))

            connection.commit()

    except Exception as e:
        app.logger.error(f"Error during embedding update: {e}")
        return jsonify({"error": str(e)})
    
    finally:
        if connection is not None:
            connection.close()

    return "Embedding Korea updated."

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host='0.0.0.0', port=port, debug=False)
