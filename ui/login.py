# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_login_Dialog(object):
    def setupUi(self, login_Dialog):
        login_Dialog.setObjectName("login_Dialog")
        login_Dialog.resize(335, 217)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/picture/computer.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        login_Dialog.setWindowIcon(icon)
        self.horizontalLayout = QtWidgets.QHBoxLayout(login_Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.titile = QtWidgets.QLabel(login_Dialog)
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(15)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.titile.setFont(font)
        self.titile.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.titile.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.titile.setTextFormat(QtCore.Qt.PlainText)
        self.titile.setAlignment(QtCore.Qt.AlignCenter)
        self.titile.setObjectName("titile")
        self.verticalLayout.addWidget(self.titile)
        self.line = QtWidgets.QFrame(login_Dialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.mid = QtWidgets.QFormLayout()
        self.mid.setObjectName("mid")
        self.user_label = QtWidgets.QLabel(login_Dialog)
        self.user_label.setObjectName("user_label")
        self.mid.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.user_label)
        self.user_line = QtWidgets.QLineEdit(login_Dialog)
        self.user_line.setObjectName("user_line")
        self.mid.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.user_line)
        self.password_label = QtWidgets.QLabel(login_Dialog)
        self.password_label.setObjectName("password_label")
        self.mid.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.password_label)
        self.password_line = QtWidgets.QLineEdit(login_Dialog)
        self.password_line.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_line.setObjectName("password_line")
        self.mid.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.password_line)
        self.verticalLayout.addLayout(self.mid)
        self.bottom = QtWidgets.QHBoxLayout()
        self.bottom.setObjectName("bottom")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.bottom.addItem(spacerItem)
        self.cancel_btn = QtWidgets.QPushButton(login_Dialog)
        self.cancel_btn.setObjectName("cancel_btn")
        self.bottom.addWidget(self.cancel_btn)
        self.signin_btn = QtWidgets.QPushButton(login_Dialog)
        self.signin_btn.setObjectName("signin_btn")
        self.bottom.addWidget(self.signin_btn)
        self.login_btn = QtWidgets.QPushButton(login_Dialog)
        self.login_btn.setObjectName("login_btn")
        self.bottom.addWidget(self.login_btn)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.bottom.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.bottom)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(login_Dialog)
        self.cancel_btn.clicked['bool'].connect(login_Dialog.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(login_Dialog)

    def retranslateUi(self, login_Dialog):
        _translate = QtCore.QCoreApplication.translate
        login_Dialog.setWindowTitle(_translate("login_Dialog", "登录"))
        self.titile.setText(_translate("login_Dialog", "MINST实验系统"))
        self.user_label.setText(_translate("login_Dialog", "用户："))
        self.password_label.setText(_translate("login_Dialog", "密码："))
        self.cancel_btn.setText(_translate("login_Dialog", "退出"))
        self.signin_btn.setText(_translate("login_Dialog", "注册"))
        self.login_btn.setText(_translate("login_Dialog", "登录"))
import resources_rc
