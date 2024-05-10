from PyQt5.QtWidgets import QApplication
from mnist_system import mnist_System
import sys
import argparse

if __name__ == '__main__':
    app = QApplication([])
    parser = argparse.ArgumentParser(description='mnist system database configuration')
    parser.add_argument('--host', type=str, default='localhost', help='host of mysql server')
    parser.add_argument('--user', type=str, default='root', help='user of mysql server')
    parser.add_argument('--password', type=str, default='123456', help='password of mysql server')
    parser.add_argument('--database', type=str, default='db_mnist_exp', help='database of mysql server')
    parser.add_argument('--login_without_info', action='store_true', default=False, help='login without user info')
    args = parser.parse_args()

    db_config = {
        'host': args.host,
        'user': args.user,
        'password': args.password,
        'database': args.database
    }

    system = mnist_System(db_config, login_without_info=args.login_without_info)
    system.show()
    sys.exit(app.exec_())
