import sqlite3

def connect_db():
    return sqlite3.connect('results.db')

def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pwd TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            PSNR FLOAT,
            CR FLOAT,
            K INTEGER NOT NULL,
            M INTEGER NOT NULL,
            height INTEGER NOT NULL,
            width INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_result(pwd, algorithm, psnr, cr, k, m, height, width):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (pwd, algorithm, PSNR, CR, K, M, height, width)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pwd, algorithm, psnr, cr, k, m, height, width))
    conn.commit()
    conn.close()

def get_all_results():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results')
    results = cursor.fetchall()
    conn.close()
    return results

def get_result_by_id(result_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results WHERE id = ?', (result_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_result(result_id, pwd, algorithm, psnr, cr, k, m, height, width):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE results
        SET pwd = ?, algorithm = ?, PSNR = ?, CR = ?, K = ?, M = ?, height = ?, width = ?
        WHERE id = ?
    ''', (pwd, algorithm, psnr, cr, k, m, height, width, result_id))
    conn.commit()
    conn.close()

def delete_result(result_alg):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results WHERE algorithm	 = ?', (result_alg,))
    conn.commit()
    conn.close()

def delete_all():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results')
    conn.commit()
    conn.close()

