from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QEvent, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QLabel, QGraphicsBlurEffect, QGraphicsOpacityEffect, QVBoxLayout


class QPicture_blur_text(QLabel):
    """
    QLabelFrame：QLabel with blur effect
    When the mouse enters the QLabelFrame, the blur effect will start.
    Hidden text will be displayed.
    And when the mouse leaves the QLabelFrame, the blur effect will disappear.
    """
    def __init__(self, image_path='', text='', parent=None):
        super(QPicture_blur_text, self).__init__()
        self.main_window = parent

        if image_path == '' or text == '':
            return
        # Create a QLabel for the image
        self.image_label = QLabel(self)
        self.image = QPixmap(image_path)
        self.image_label.setPixmap(self.image)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Create a QLabel for the text
        self.text_label = QLabel(self)
        self.text_label.setText(text)
        self.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setFont(QFont("Cascadia Code", 12))
        self.text_label.hide()

        # Create a QPropertyAnimation for the blur effect
        self.blur_effect_obj = QGraphicsBlurEffect()

        self.blur_animation = QPropertyAnimation(self.blur_effect_obj, b"blurRadius")
        self.blur_animation.setDuration(500)
        self.blur_animation.setStartValue(0)
        self.blur_animation.setEndValue(10)
        self.blur_animation.setEasingCurve(QEasingCurve.OutQuad)

        # Create a QPropertyAnimation for the text color
        self.text_effect_obj = QGraphicsOpacityEffect()
        self.text_animation = QPropertyAnimation(self.text_effect_obj, b"opacity")
        self.text_animation.setDuration(500)
        self.text_animation.setStartValue(0)
        self.text_animation.setEndValue(1)
        self.text_animation.setEasingCurve(QEasingCurve.OutQuad)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_label)

        self.text_label.resize(self.size())

    def enterEvent(self, event: QEvent):
        self.text_label.show()
        self.text_label.setGraphicsEffect(self.text_effect_obj)
        self.image_label.setGraphicsEffect(self.blur_effect_obj)

        self.blur_animation.setDirection(QPropertyAnimation.Forward)
        self.blur_animation.start()
        self.text_animation.setDirection(QPropertyAnimation.Forward)
        self.text_animation.start()

    def leaveEvent(self, event: QEvent):
        self.blur_animation.setDirection(QPropertyAnimation.Backward)
        self.blur_animation.start()
        self.text_animation.setDirection(QPropertyAnimation.Backward)
        self.text_animation.start()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.text_label.resize(self.size())

    def change_text(self, text):
        self.text_label.setText(text)
