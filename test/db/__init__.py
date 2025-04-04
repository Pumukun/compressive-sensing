from .db import (
    connect_db,
    create_table,
    add_result,
    get_all_results,
    get_result_by_id,
    update_result,
    delete_result,
    delete_all,
    get_result_by_alg
)

__all__ = [
    'connect_db',
    'create_table',
    'add_result',
    'get_all_results',
    'get_result_by_id',
    'update_result',
    'delete_result',
    'delete_all'
    'get_result_by_alg'
]
