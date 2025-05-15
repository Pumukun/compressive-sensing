import sqlite3
from typing import List, Tuple, Optional, Union

'''by Mikhail Shibanov'''

def connect_db() -> sqlite3.Connection:
    '''Подключение к базе данных SQLite results.db'''
    return sqlite3.connect('results.db')

def create_table() -> None:
    '''Создание таблицы results, если она не существует'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_image TEXT NOT NULL,
            pwd TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            PSNR FLOAT,
            SSIM FLOAT,
            CR FLOAT,
            K INTEGER NOT NULL,
            M INTEGER NOT NULL,
            height INTEGER NOT NULL,
            width INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_result(
    pwd: str,
    original_image: str,
    algorithm: str,
    psnr: Optional[float],
    ssim: Optional[float],
    cr: Optional[float],
    k: int,
    m: int,
    height: int,
    width: int
) -> None:
    '''Добавление новой записи в таблицу results'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (pwd, original_image, algorithm, PSNR, SSIM, CR, K, M, height, width)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pwd, original_image, algorithm, psnr, ssim, cr, k, m, height, width))
    conn.commit()
    conn.close()

def get_all_results() -> List[Tuple]:
    '''Получение всех записей из таблицы results'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results')
    results = cursor.fetchall()
    conn.close()
    return results

def get_result_by_id(result_id: int) -> Optional[Tuple]:
    '''Получение записи по ID'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results WHERE id = ?', (result_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_result_by_alg(alg: str) -> List[Tuple]:
    '''Получение всех записей для указанного алгоритма'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results WHERE algorithm = ?', (alg,))
    result = cursor.fetchall()
    conn.close()
    return result

def update_result(
    result_id: int,
    pwd: str,
    algorithm: str,
    psnr: Optional[float],
    cr: Optional[float],
    k: int,
    m: int,
    height: int,
    width: int
) -> None:
    '''Обновление записи по её ID'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE results
        SET pwd = ?, algorithm = ?, PSNR = ?, CR = ?, K = ?, M = ?, height = ?, width = ?
        WHERE id = ?
    ''', (pwd, algorithm, psnr, cr, k, m, height, width, result_id))
    conn.commit()
    conn.close()

def delete_result(result_alg: str) -> None:
    '''Удаление всех записей для указанного алгоритма'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results WHERE algorithm = ?', (result_alg,))
    conn.commit()
    conn.close()

def delete_all() -> None:
    '''Удаление всех записей из таблицы results'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results')
    conn.commit()
    conn.close()
