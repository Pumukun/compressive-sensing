import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Sequence, Union

# Absolute, so the database does not depend on the current directory
DB_PATH: Path = Path(
    os.environ.get('CS_RESULTS_DB', Path(__file__).resolve().parent.parent / 'results.db')
)

_INSERT_SQL = '''
    INSERT INTO results (pwd, original_image, algorithm, PSNR, SSIM, CR, K, M, height, width)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''


def connect_db() -> sqlite3.Connection:
    '''Connect to the SQLite database results.db'''
    conn = sqlite3.connect(DB_PATH)
    # WAL: readers do not block the writer during parallel runs
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def create_table() -> None:
    '''Create the results table if it does not exist'''
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
    '''Insert a single row into the results table'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(_INSERT_SQL, (pwd, original_image, algorithm, psnr, ssim, cr, k, m, height, width))
    conn.commit()
    conn.close()

def add_results(rows: Iterable[Sequence]) -> None:
    '''
    Batched insert. Each row is a tuple
    (pwd, original_image, algorithm, psnr, ssim, cr, k, m, height, width).
    '''
    rows = list(rows)
    if not rows:
        return

    conn = connect_db()
    try:
        conn.executemany(_INSERT_SQL, rows)
        conn.commit()
    finally:
        conn.close()

def get_all_results() -> List[Tuple]:
    '''Fetch every row from the results table'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results')
    results = cursor.fetchall()
    conn.close()
    return results

def get_result_by_id(result_id: int) -> Optional[Tuple]:
    '''Fetch a single row by ID'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM results WHERE id = ?', (result_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_result_by_alg(alg: str) -> List[Tuple]:
    '''Fetch every row for the given algorithm'''
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
    '''Update a row by its ID'''
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
    '''Delete every row for the given algorithm'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results WHERE algorithm = ?', (result_alg,))
    conn.commit()
    conn.close()

def delete_all() -> None:
    '''Delete every row from the results table'''
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM results')
    conn.commit()
    conn.close()
