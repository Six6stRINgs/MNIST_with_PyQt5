import numpy
import torch
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QTextCursor, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QMainWindow, QMessageBox, QTableWidgetItem, QHeaderView, QTableWidget, \
    QFileDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis

from Qt_class.TrainAndTestThread import TrainAndTestThread
from Qt_class.QPicture_blur_text import QPicture_blur_text

from ui.login import Ui_login_Dialog
from ui.mw import Ui_MainWindow
from ui.usr_setting import Ui_Setting_Form

from utils import sha256, check_user_exist, get_authority_by_usr, get_all_usr_id, get_usr_info_by_id, \
    search_model_file, get_all_file_name, get_info_from_saved_model, add_new_pic_to_display_widget

from data import load_custom_image, np2QPixmap

import netron
import onnxruntime
import os

'''
Q: Why I am not using the QSqlDatabase, instead of using mysql.connector directly?
A: QSqlDatabase is a class provided by Qt designed for database connection. It's convenient for Qt-based project.
   Unfortunately, some denpendencies are missing in my PyQt5 package when I am using the MySQL driver.
   It's a similar issue, which I spent a lot of time trying to solve it 1 year ago when I was working 
   on other project using C++.
   Indeed I solved it by compiling the MySQL denpendencies by myself, but it's too complicated for this assignment.
   For a simple project like this, I think mysql.connector is enough.
'''
from mysql.connector.abstracts import MySQLCursorAbstract
import mysql.connector


class mnist_System:
    def __init__(self, db_config: dict = None, login_without_info: bool = False):
        super().__init__()
        self.first_load = True
        self.usr = 'admin'
        self.password = '__NO_PASSWORD__'
        self.usr_authority = True
        self.login_without_info = login_without_info

        self.login_ori_ui = Ui_login_Dialog()
        self.mw_ori_ui = Ui_MainWindow()

        self.login_dialog = QDialog()
        self.mw = QMainWindow()

        self.tmp_wid = None
        self.web_view = None
        # Train need a long time, so we need a thread to do this to avoid blocking the UI
        self.train_thread = None

        # loss plot
        self.loss_series = QLineSeries()
        self.loss_chart = QChart()
        self.loss_chart_view = QChartView(self.loss_chart)
        self.max_loss = 0.0

        self.model_path = 'model'

        self.login_ori_ui.setupUi(self.login_dialog)
        self.mw_ori_ui.setupUi(self.mw)

        if db_config is not None:
            self.db = mysql.connector.connect(**db_config)

        self.model_config = {
            'model': 'None',
            'epochs': 0,
            'batch_size': 0,
            'lr': 0.0,
            'loss': 'None',
            'optimizer': 'None',
            'dropout': 0.0,
            'device': 'None',
            'user': self.usr,
            'test_acc': 0.0,
            'model_type': 'None'
        }

        # classification rate of 0~9 numbers
        self.cls_rate_num_list = [getattr(self.mw_ori_ui, f'p{i}_line') for i in range(10)]
        self.cls_rate_lcd_list = [getattr(self.mw_ori_ui, f'p{i}_lcdNumber') for i in range(10)]
        self.cls_confirm_btn_list = [getattr(self.mw_ori_ui, f'p{i}_btn') for i in range(10)]

        self.last_file_dialog_path = './'
        self.model_str_label = self.__get_model_str_label_from_model_config(no_info=True)
        self.exist_model_save_name = []
        self.cur_cla_model = None
        self.cls_input_image_path = None
        self.cls_pic_labelframe = QPicture_blur_text()
        # classification accuracy
        self.acc_top1 = 0
        self.acc_top3 = 0
        self.acc_top5 = 0
        self.total_num = 0

        self.__login_init()
        self.__mw_init()
        self.first_load = False

    def show(self):
        self.login_dialog.show()

    def __login_init(self):
        self.login_dialog.setWhatsThis("Login dialog for mnist system. \n"
                                       "An assignment of software engineering. \n"
                                       "Author: a CS21-1 student from NCUT.")
        self.login_ori_ui.login_btn.clicked.connect(self.__login_check)
        self.login_ori_ui.signin_btn.clicked.connect(lambda:
                                                     self.__signin(self.login_dialog,
                                                                   self.login_ori_ui.user_line.text(),
                                                                   self.login_ori_ui.password_line.text()
                                                                   )
                                                     )

    def __login_check(self):
        user, password = self.login_ori_ui.user_line.text(), self.login_ori_ui.password_line.text()
        cursor = self.db.cursor()
        if self.login_without_info or user and password and check_user_exist(cursor, user, password):
            self.login_dialog.close()
            if not self.login_without_info:
                self.usr = user
                self.password = password
                self.usr_authority = get_authority_by_usr(cursor, user)
            self.mw_ori_ui.user_top_label.setText(f"Welcome, {self.usr}! ")
            self.mw.show()
            print(f"Login successful, Direct Login: {self.login_without_info} "
                  f",user: {self.usr}, password: {self.password}")
        else:
            QMessageBox.critical(self.login_dialog, "Login failed", "User not found or password incorrect")
            print(f"Login failed, user: {user}, password: {password}")

    def __signin(self, wid: QWidget, user: str, password: str, authority: bool = False):
        cursor = self.db.cursor()
        if user and check_user_exist(cursor, user):
            QMessageBox.critical(wid, "Register failed", "User already exists")
            print(f"User already exists, user: {user}, password: {password}")
        elif user and password:
            sql = f"INSERT INTO user(name, password, authority) VALUES('{user}', '{sha256(password)}', {authority})"
            cursor.execute(sql)
            self.db.commit()
            QMessageBox.information(wid, "Register Successful", "User created")
            print(f"User created, SQL: {sql}")
        else:
            QMessageBox.critical(wid, "Register failed", "User or password cannot be empty")
            print(f"Register failed, user: {user}, password: {password}")

    def __change_usr_info(self, wid: QWidget, user_id: str, user: str, password: str, authority: bool):
        cursor = self.db.cursor()
        if user and check_user_exist(cursor, user):
            QMessageBox.critical(wid, "Update failed", "User already exists")
            print(f"User already exists, user: {user}, password: {password}")
        elif user and password:
            sql = f"UPDATE user SET name='{user}', password='{sha256(password)}', " \
                  f"authority={authority} WHERE user_id={user_id}"
            cursor.execute(sql)
            self.db.commit()
            QMessageBox.information(wid, "Update Successful", "User info updated")
            print(f"User info updated, SQL: {sql}")
        else:
            QMessageBox.critical(wid, "Update failed", "User or password cannot be empty")
            print(f"Update failed, user: {user}, password: {password}")

    def __mw_init(self):
        self.mw.resize(1200, 900)
        self.mw_ori_ui.tabWidget.setCurrentIndex(0)
        self.mw_ori_ui.re_login.triggered.connect(lambda: (
            self.mw.close(),
            self.__init__(),
            self.login_dialog.show()
        ))
        # Init about user management
        self.__mw_usr_tablewidget_update()
        self.mw_ori_ui.create_usr_btn.clicked.connect(self.__mw_create_usr)
        self.mw_ori_ui.select_usr_btn.clicked.connect(self.__mw_select_usr)
        self.mw_ori_ui.update_usr_btn.clicked.connect(self.__mw_update_usr)
        self.mw_ori_ui.delete_usr_btn.clicked.connect(self.__mw_delete_usr)
        self.mw_ori_ui.refresh_usr_btn.clicked.connect(lambda: (
            self.__mw_usr_tablewidget_update(),
            print("Refresh user tablewidget")
        ))
        # Init about model setting
        self.mw_ori_ui.train_progressBar.hide()
        self.mw_ori_ui.model_tabWidget.setCurrentIndex(0)
        self.mw_ori_ui.model_left_tabWidget.setCurrentIndex(0)
        if not torch.cuda.is_available():
            self.mw_ori_ui.device_combo.removeItem(0)
        # Disable the second tab (loss plot)
        # Enable it when the model is training
        self.mw_ori_ui.model_left_tabWidget.setTabEnabled(1, False)
        self.__mw_model_visualizer_broswer_init()
        self.__mw_model_loss_plot_init()
        self.mw_ori_ui.train_model_btn.clicked.connect(self.__mw_model_begin_train)
        # When the model combo box changed, refresh the browser
        self.mw_ori_ui.model_combo.currentTextChanged.connect(lambda text: (
            d_dspin := self.mw_ori_ui.dropout_dspin,
            vis_label_text := "Visualizer:",
            # ViT model does not support ONNX format
            # So Visulizer is not supported for ViT model
            self.mw_ori_ui.vis_label.setText(
                vis_label_text if text != 'ViT' else vis_label_text[:-1] + ' (Unsupport for ViT):'
            ),
            d_dspin.setEnabled(
                False if text == 'ResNet18' or text == 'ResNet34' or text == 'ViT' else True
            ),
            self.__mw_model_visualizer_broswer_init()
        ))
        # Init about classification widget
        self.cls_pic_labelframe = QPicture_blur_text("./qrc/network.jpg", self.model_str_label)
        self.mw_ori_ui.network_widget.layout().addWidget(self.cls_pic_labelframe)
        self.__update_saved_model_combo()
        self.mw_ori_ui.input_btn.clicked.connect(self.__get_picture_from_file_dialog)
        self.mw_ori_ui.cls_begin_btn.clicked.connect(self.__begin_classify)
        # Init about classification confirm button
        for i, btn in enumerate(self.cls_confirm_btn_list):
            # click event has a bool parameter, so we need to use lambda to pass the parameter
            # then we can use the index of the button to confirm the classification
            btn.clicked.connect(lambda checked, i=i: self.__confirm_classify(i))

    def __mw_usr_tablewidget_update(self, usr_id: int = -1, usr: str = '', password: str = '', authority: int = -1):
        cursor = self.db.cursor()
        condition = []
        if usr_id != -1:
            condition.append(f"user_id={usr_id}")
        if usr:
            condition.append(f"name='{usr}'")
        if password:
            condition.append(f"password='{sha256(password)}'")
        if authority != -1:
            condition.append(f"authority={authority}")

        if condition:
            condition = ' AND '.join(condition)
            print(f"tablewidget update condition: {condition}")
            cursor.execute(f"SELECT * FROM user WHERE {condition}")
        else:
            cursor.execute("SELECT * FROM user")
        res = cursor.fetchall()
        self.mw_ori_ui.usr_tablewidget.setRowCount(len(res))
        self.mw_ori_ui.usr_tablewidget.setColumnCount(len(res[0]))
        self.mw_ori_ui.usr_tablewidget.setHorizontalHeaderLabels(['User Id', 'Name', 'Password', 'Authority'])
        self.mw_ori_ui.usr_tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mw_ori_ui.usr_tablewidget.setEditTriggers(QTableWidget.NoEditTriggers)
        for i, row in enumerate(res):
            for j, cell in enumerate(row):
                if j == 3:
                    cell = 'Normal' if cell == 0 else 'Administrator'
                self.mw_ori_ui.usr_tablewidget.setItem(i, j, QTableWidgetItem(str(cell)))

    def __mw_create_usr(self):
        if not self.__permission_check():
            return
        self.tmp_wid = QDialog()
        self.tmp_wid.setWhatsThis("Create a new user")
        setting_ori_ui = Ui_Setting_Form()
        setting_ori_ui.setupUi(self.tmp_wid)
        setting_ori_ui.id_combo.hide()
        setting_ori_ui.id_label.setText("新用户设置:")
        setting_ori_ui.yes_btn.clicked.connect(lambda: (
            self.__signin(self.tmp_wid,
                          setting_ori_ui.user_line.text(),
                          setting_ori_ui.password_line.text(),
                          setting_ori_ui.authority_combo.currentIndex()
                          ),
            self.__mw_usr_tablewidget_update(),
            self.tmp_wid.close()
        ))
        self.tmp_wid.show()

    def __mw_select_usr(self):
        self.tmp_wid = QDialog()
        self.tmp_wid.setWhatsThis("Select certain user info. \n"
                                  "Empty line means no condition.")
        setting_ori_ui = Ui_Setting_Form()
        setting_ori_ui.setupUi(self.tmp_wid)
        setting_ori_ui.id_combo.addItems([str(i) for i in (get_all_usr_id(self.db.cursor())) + ['ALL']])
        setting_ori_ui.id_combo.setCurrentIndex(len(setting_ori_ui.id_combo) - 1)
        setting_ori_ui.authority_combo.addItem("ALL")
        setting_ori_ui.authority_combo.setCurrentIndex(len(setting_ori_ui.authority_combo) - 1)
        setting_ori_ui.yes_btn.clicked.connect(lambda: (
            id_sel := setting_ori_ui.id_combo.currentText(),
            authority_sel := setting_ori_ui.authority_combo.currentIndex(),
            self.__mw_usr_tablewidget_update(
                int(id_sel) if id_sel != 'ALL' else -1,
                setting_ori_ui.user_line.text(),
                setting_ori_ui.password_line.text(),
                authority_sel if authority_sel < 2 else -1
            ),
            self.tmp_wid.close()
        ))
        self.tmp_wid.show()

    def __mw_update_usr(self):
        if not self.__permission_check():
            return
        self.tmp_wid = QDialog()
        self.tmp_wid.setWhatsThis("Update certain user info \n"
                                  "User id cannot be changed.")
        setting_ori_ui = Ui_Setting_Form()
        setting_ori_ui.setupUi(self.tmp_wid)
        setting_ori_ui.id_combo.addItems([str(i) for i in get_all_usr_id(self.db.cursor())])
        setting_ori_ui.id_combo.currentIndexChanged.connect(lambda: (
            setting_ori_ui.user_line.setText(
                get_usr_info_by_id(self.db.cursor(),
                                   int(setting_ori_ui.id_combo.currentText()))['name'])
        ))
        setting_ori_ui.yes_btn.clicked.connect(lambda: (
            self.__change_usr_info(self.tmp_wid,
                                   setting_ori_ui.id_combo.currentText(),
                                   setting_ori_ui.user_line.text(),
                                   setting_ori_ui.password_line.text(),
                                   setting_ori_ui.authority_combo.currentIndex()
                                   ),
            self.__mw_usr_tablewidget_update(),
            self.tmp_wid.close()
        ))

        self.tmp_wid.show()

    def __mw_delete_usr(self):
        if not self.__permission_check():
            return
        if not self.mw_ori_ui.usr_tablewidget.selectedItems():
            QMessageBox.critical(self.mw, "Delete", "Please select a row to delete")
            return
        reply = QMessageBox.information(self.mw, "Delete", "Confirm to delete the selected row?",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            cursor = self.db.cursor()
            for item in self.mw_ori_ui.usr_tablewidget.selectedItems():
                item_id = self.mw_ori_ui.usr_tablewidget.item(item.row(), 0).text()
                cursor.execute(f"DELETE FROM user WHERE user_id={item_id}")
            self.db.commit()
            self.__mw_usr_tablewidget_update()

    def __permission_check(self) -> bool:
        if self.usr_authority:
            return True
        else:
            QMessageBox.critical(self.mw, "Permission denied", "You have no permission to do this operation")
            return False

    def __mw_model_visualizer_broswer_init(self):
        model_name = self.mw_ori_ui.model_combo.currentText()
        if model_name != 'ViT':
            model_path = search_model_file(self.model_path, model_name)
        else:
            model_path = ''
        netron.stop()
        netron.start(model_path, browse=False)
        self.mw_ori_ui.vis_wid.layout().removeWidget(self.web_view)
        self.web_view = QWebEngineView()
        self.mw_ori_ui.vis_wid.layout().addWidget(self.web_view)
        # website of netron
        self.web_view.load(QUrl("http://localhost:8080"))
        self.mw_ori_ui.refresh_brower_btn.clicked.connect(lambda: (
            self.web_view.reload(),
            print("Refresh browser")
        ))

    def __mw_model_begin_train(self):
        self.__get_model_config(model_name='', change_text=False)
        print(self.model_config)
        reply = QMessageBox.information(self.mw, "Train", "Confirm to start training at this configuration?"
                                                          "Classify widget will be reset.",
                                        QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.No:
            return

        self.mw_ori_ui.train_progressBar.show()
        self.mw_ori_ui.model_left_tabWidget.setTabEnabled(1, True)
        self.mw_ori_ui.model_left_tabWidget.setCurrentIndex(1)

        self.train_thread = TrainAndTestThread(self.model_config)
        self.train_thread.progress_signal.connect(self.mw_ori_ui.train_progressBar.setValue)
        self.train_thread.text_signal.connect(lambda text: (
            self.mw_ori_ui.train_textinfo.append(text),
            self.mw_ori_ui.train_textinfo.moveCursor(QTextCursor.End)
        ))
        self.train_thread.loss_signal.connect(self.__update_loss_plot)
        self.train_thread.finished.connect(self.__on_train_finished)
        self.train_thread.start()

    def __on_train_finished(self):
        QMessageBox.information(self.mw, "Train", "Training finished")
        self.mw_ori_ui.train_progressBar.hide()
        self.__mw_model_visualizer_broswer_init()
        self.__update_saved_model_combo()
        print("Refresh browser")

    def __mw_model_loss_plot_init(self):
        self.mw_ori_ui.loss_plot_wid.layout().addWidget(self.loss_chart_view)
        self.loss_series.clear()
        self.loss_chart.removeAllSeries()
        self.loss_chart.setAnimationOptions(QChart.SeriesAnimations)
        self.loss_chart.setTheme(QChart.ChartThemeDark)
        self.loss_chart.legend().hide()
        self.loss_chart.setTitleFont(self.mw_ori_ui.train_textinfo.font())
        self.loss_chart_view.setChart(self.loss_chart)
        self.loss_chart_view.setRenderHint(QPainter.Antialiasing)

    def __update_loss_plot(self, loss: float):
        self.loss_series.append(self.loss_series.count(), loss)
        self.max_loss = max(self.max_loss, loss)
        self.loss_chart.addSeries(self.loss_series)

        axisX = QValueAxis()
        axisY = QValueAxis()
        axisX.setRange(0, self.loss_series.count())
        axisX.setTickInterval(1)
        axisX.setLabelFormat("%d")
        axisY.setRange(0, self.max_loss * 1.2)
        # add axis to chart
        self.loss_chart.setAxisX(axisX, self.loss_series)
        self.loss_chart.setAxisY(axisY, self.loss_series)
        self.loss_chart.axisX().setTitleText("Step")
        self.loss_chart.axisY().setTitleText("Loss")
        self.loss_chart.setTitle(f"Lastest loss: {loss:.4f}")

    def __get_picture_from_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self.mw, "Open Image", self.last_file_dialog_path,
                                                   "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.cls_input_image_path = file_name
            self.last_file_dialog_path = os.path.dirname(file_name)
            add_new_pic_to_display_widget(QPixmap(self.cls_input_image_path), self.mw_ori_ui.input_pic_widget)

    def __load_model(self, model_name: str):
        if model_name == 'None':
            self.__mw_cls_number_setting_reset()
            self.model_str_label = self.__get_model_str_label_from_model_config(no_info=True)
            self.cls_pic_labelframe.change_text(self.model_str_label)
            self.cur_cla_model = None
            return
        if self.mw_ori_ui.p0_line.text() != '':  # first time without confirm
            reply = QMessageBox.information(self.mw, "Load model", "Confirm to load this model? \n"
                                                                   "Current result will be reset.",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            reply = QMessageBox.Yes
        if reply == QMessageBox.Yes:
            self.mw_ori_ui.model_tabWidget.setCurrentIndex(1)
            model_path = os.path.join(self.model_path, model_name)
            self.__get_model_config(model_name)
            if self.model_config['model_type'] == 'ONNX':
                print(f"Load ONNX model: {model_name}")
                self.cur_cla_model = onnxruntime.InferenceSession(model_path)
            elif self.model_config['model_type'] == 'PT':
                print(f"Load PT model: {model_name}")
                self.cur_cla_model = torch.load(model_path)
            self.__mw_cls_number_setting_reset()

    def __mw_cls_number_setting_reset(self):
        for line in self.cls_rate_num_list:
            line.setText('')
        for lcd in self.cls_rate_lcd_list:
            lcd.setStyleSheet("QLCDNumber {color: black;}")
            lcd.setStyleSheet("QLineEdit {color: black;}")

        self.mw_ori_ui.sel_label_lcdNumber.display(-1)

        for btn in self.cls_confirm_btn_list:
            btn.setEnabled(True)

        self.acc_top1, self.acc_top3, self.acc_top5 = 0, 0, 0
        self.total_num = 0

        self.mw_ori_ui.acc_top1_num.setText("")
        self.mw_ori_ui.acc_top3_num.setText("")
        self.mw_ori_ui.acc_top5_num.setText("")

    def __begin_classify(self):
        if self.cur_cla_model is None:
            QMessageBox.critical(self.mw, "Classify", "Please select a model first")
            return
        if self.cls_input_image_path is None:
            QMessageBox.critical(self.mw, "Classify", "Please select an image first")
            return
        # prepare the input image
        img = load_custom_image(self.cls_input_image_path)
        add_new_pic_to_display_widget(np2QPixmap(img), self.mw_ori_ui.progress_pic_widget)

        # begin to classify
        if self.model_config['model_type'] == 'ONNX':
            input_name = self.cur_cla_model.get_inputs()[0].name
            output_name = self.cur_cla_model.get_outputs()[0].name
            output = self.cur_cla_model.run([output_name], {input_name: img})[0]
            ouput = numpy.exp(output) / numpy.sum(numpy.exp(output))  # softmax
        elif self.model_config['model_type'] == 'PT':
            img = torch.from_numpy(img).unsqueeze(0).float().to(self.model_config['device'])
            self.cur_cla_model.to(self.model_config['device'])
            self.cur_cla_model.eval()
            with torch.no_grad():
                output = self.cur_cla_model(img)
                ouput = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        print("Output: ", ouput)
        self.output = ouput

        for i, rate in enumerate(ouput[0]):
            self.cls_rate_num_list[i].setText(f"{rate:.4f}")

        max_rate = max(ouput[0])
        for i, rate in enumerate(ouput[0]):
            if rate == max_rate:
                self.cls_rate_lcd_list[i].setStyleSheet("QLCDNumber {color: red;}")
                self.cls_rate_num_list[i].setStyleSheet("QLineEdit {color: red;}")
            else:
                self.cls_rate_lcd_list[i].setStyleSheet("QLCDNumber {color: black;}")
                self.cls_rate_num_list[i].setStyleSheet("QLineEdit {color: black;}")

        for btn in self.cls_confirm_btn_list:
            btn.setEnabled(True)

    def __confirm_classify(self, num: int):
        self.mw_ori_ui.sel_label_lcdNumber.display(num)
        for i, btn in enumerate(self.cls_confirm_btn_list):
            if btn.hasFocus():
                # Move focus to another widget
                self.mw.setFocus()
            btn.setEnabled(False)

        sort_index = numpy.argsort(self.output[0])
        if num in sort_index[-5:]:  # top5
            self.acc_top5 += 1
        if num in sort_index[-3:]:  # top3
            self.acc_top3 += 1
        if num == sort_index[-1]:  # top1
            self.acc_top1 += 1

        self.total_num += 1
        self.mw_ori_ui.acc_top1_num.setText(f"{(self.acc_top1 / self.total_num) * 100:.2f}%")
        self.mw_ori_ui.acc_top3_num.setText(f"{(self.acc_top3 / self.total_num) * 100:.2f}%")
        self.mw_ori_ui.acc_top5_num.setText(f"{(self.acc_top5 / self.total_num) * 100:.2f}%")

    def __get_model_str_label_from_model_config(self, no_info: bool = False):
        if no_info:
            return 'Network Setting \n' \
                   'Model name: None \n' \
                   'Epochs: None \n' \
                   'Batch size: None \n' \
                   'Learning rate: None \n' \
                   'Loss: None \n' \
                   'Optimizer: None \n' \
                   'Dropout: None \n' \
                   'Device: None \n' \
                   'Author: None \n' \
                   'Test accuracy: None \n' \
                   'Model type: None'
        else:
            return 'Network Setting \n' \
                   'Model name: {} \n' \
                   'Epochs: {} \n' \
                   'Batch size: {} \n' \
                   'Learning rate: {} \n' \
                   'Loss: {} \n' \
                   'Optimizer: {} \n' \
                   'Dropout: {} \n' \
                   'Device: {} \n' \
                   'Author: {} \n' \
                   'Test accuracy: {}% \n'\
                   'Model type: {}'.format(
                        self.model_config['model'],
                        self.model_config['epochs'],
                        self.model_config['batch_size'],
                        self.model_config['lr'],
                        self.model_config['loss'],
                        self.model_config['optimizer'],
                        self.model_config['dropout'],
                        self.model_config['device'],
                        self.model_config['user'],
                        self.model_config['test_acc'],
                        self.model_config['model_type']
                   )

    def __get_model_config(self, model_name: str = '', change_text: bool = True):
        if model_name == '':
            self.model_config['model'] = self.mw_ori_ui.model_combo.currentText()
            self.model_config['epochs'] = self.mw_ori_ui.epoch_spin.value()
            self.model_config['batch_size'] = self.mw_ori_ui.batch_spin.value()
            self.model_config['lr'] = self.mw_ori_ui.lr_dspin.value()
            self.model_config['loss'] = self.mw_ori_ui.loss_combo.currentText()
            self.model_config['optimizer'] = self.mw_ori_ui.opt_combo.currentText()
            self.model_config['dropout'] = self.mw_ori_ui.dropout_dspin.value()
            self.model_config['device'] = 'cuda' if self.mw_ori_ui.device_combo.currentText() == 'GPU' else 'cpu'
            self.model_config['user'] = self.usr
            if str(self.model_config['model']) != 'ViT':
                self.model_config['model_type'] = 'ONNX'
            else:
                self.model_config['model_type'] = 'PT'
        else:
            self.model_config = get_info_from_saved_model(model_name)

        self.model_str_label = self.__get_model_str_label_from_model_config()
        if change_text:
            self.cls_pic_labelframe.change_text(self.model_str_label)

        # print('New model config: ', self.model_config)
        # print('New model str label: ', self.model_str_label)

    def __update_saved_model_combo(self):
        self.exist_model_save_name = get_all_file_name('model')
        if not self.first_load:
            print("Disconnect signal")
            self.mw_ori_ui.cls_model_sel_combo.currentTextChanged.disconnect()
        self.mw_ori_ui.cls_model_sel_combo.clear()
        self.mw_ori_ui.cls_model_sel_combo.addItems([name for name in ["None"] + self.exist_model_save_name])
        self.mw_ori_ui.cls_model_sel_combo.currentTextChanged.connect(lambda text: (
            print("Begin to load model"),
            self.__load_model(text)
        ))
        self.__mw_cls_number_setting_reset()
