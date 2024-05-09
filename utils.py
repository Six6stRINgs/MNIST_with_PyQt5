import hashlib
import os
import re

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel
from mysql.connector.abstracts import MySQLCursorAbstract


def sha256(s: str, print_flag: bool = True) -> str:
    res = hashlib.sha256(s.encode()).hexdigest()
    if print_flag:
        print(f"sha256({s}) = {res}")
    return res


def check_user_exist(cursor: MySQLCursorAbstract, user: str, password: str = '') -> bool:
    if password:
        cursor.execute(f"SELECT * FROM user WHERE name='{user}' AND password='{sha256(password, False)}'")
    else:
        cursor.execute(f"SELECT * FROM user WHERE name='{user}'")
    return bool(cursor.fetchone())


def get_all_usr_id(cursor: MySQLCursorAbstract) -> list:
    cursor.execute("SELECT user_id FROM user")
    return [i[0] for i in cursor.fetchall()]


def get_usr_info_by_id(cursor: MySQLCursorAbstract, id_: int) -> dict:
    cursor.execute(f"SELECT * FROM user WHERE user_id={id_}")
    res = cursor.fetchone()
    return {
        'user_id': res[0],
        'name': res[1],
        'password': res[2],
        'authority': res[3]
    }


def get_authority_by_usr(cursor: MySQLCursorAbstract, user: str) -> int:
    cursor.execute(f"SELECT authority FROM user WHERE name='{user}'")
    return cursor.fetchone()[0]


def search_model_file(model_path: str, model_name: str) -> str:
    file_names = os.listdir(model_path)
    for file_name in file_names:
        if re.match(f".*{model_name}.*", file_name):
            return os.path.join(model_path, file_name)
    return ""


def get_all_file_name(file_path: str) -> list[str]:
    return os.listdir(f"./{file_path}")


def get_info_from_saved_model(filename: str) -> dict:
    pattern = r"Model_(.*)_Epochs_(.*)_Batch_(.*)_Lr_(.*)_Loss_(.*)_Opt_(.*)_Dropout_(.*)" \
              r"_Device_(.*)_Author_(.*)_Acc_(.*)_.(.*)"
    match = re.match(pattern, filename)
    if match:
        return {
            'model': match.group(1),
            'epochs': match.group(2),
            'batch_size': match.group(3),
            'lr': match.group(4),
            'loss': match.group(5),
            'optimizer': match.group(6),
            'dropout': match.group(7),
            'device': match.group(8),
            'user': match.group(9),
            'test_acc': match.group(10),
            'model_type': match.group(11).upper()
        }
    else:
        return {
            'model': 'None',
            'epochs': 'None',
            'batch_size': 'None',
            'lr': 'None',
            'loss': 'None',
            'optimizer': 'None',
            'dropout': 'None',
            'device': 'None',
            'user': 'None',
            'test_acc': 'None',
            'model_type': 'None'
        }


def add_new_pic_to_display_widget(picture: QPixmap, widget: QWidget):
    widget.layout().removeItem(widget.layout().itemAt(0))
    pic_label = QLabel()
    pic_label.setPixmap(picture)
    pic_label.setScaledContents(True)
    pic_label.setAlignment(Qt.AlignCenter)
    widget.layout().addWidget(pic_label)
