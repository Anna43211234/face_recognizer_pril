import sys
import cv2
import numpy as np
import os
import json
import threading
import time
from datetime import datetime

# Импорты PySide6 для построения графического интерфейса
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QInputDialog, QListWidgetItem)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QTimer, Qt, QFile, QThread, Signal, QMutex
from PySide6.QtGui import QImage, QPixmap, QFont

# Для синтеза речи (Google TTS) и воспроизведения через pygame
from gtts import gTTS
import pygame
import tempfile

# Для работы с MySQL
import mysql.connector
from mysql.connector import Error


# Класс для управления подключением и запросами к базе данных
class DatabaseManager:

    def __init__(self):
        self.connection = None
        self.connect()  # сразу устанавливаем соединение

    def connect(self):
        # Устанавливает соединение с MySQL сервером
        try:
            self.connection = mysql.connector.connect(
                host='MySQL-8.4',          # хост (может отличаться в вашей среде)
                database='face_recognition_db',
                user='root',
                password=''
            )
            if self.connection.is_connected():
                print("Успешное подключение к базе данных")
        except Error as e:
            print(f"Ошибка подключения к базе данных: {e}")
            self.connection = None

    def log_recognition(self, username, confidence):
        # Добавляет запись о распознавании в таблицу recognition_logs
        # Возвращает True при успехе, иначе False
        if self.connection is None:
            print("Нет подключения к базе данных")
            return False

        try:
            cursor = self.connection.cursor()
            query = """INSERT INTO recognition_logs (username, confidence, recognition_time) 
                       VALUES (%s, %s, %s)"""
            recognition_time = datetime.now()
            cursor.execute(query, (username, confidence, recognition_time))
            self.connection.commit()
            cursor.close()
            print(f"Запись добавлена: {username} распознан с уверенностью {confidence}%")
            return True
        except Error as e:
            print(f"Ошибка при записи в базу данных: {e}")
            return False

    def get_recent_logs(self, limit=50):
        # Возвращает последние limit записей из таблицы логов
        if self.connection is None:
            return []

        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """SELECT * FROM recognition_logs 
                       ORDER BY recognition_time DESC 
                       LIMIT %s"""
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"Ошибка при чтении из базы данных: {e}")
            return []

    def close(self):
        # Закрывает соединение с БД
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Соединение с базой данных закрыто")


# Поток для захвата видео с веб-камеры в фоновом режиме
class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)  # сигнал для передачи кадра в главный поток

    def __init__(self):
        super().__init__()
        self._run_flag = True   # управляет циклом захвата
        self.mutex = QMutex()   # защита флага от гонок

    def run(self):
        # Запускает цикл чтения кадров с камеры
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)  # отправляем кадр
            self.msleep(30)  # пауза для снижения нагрузки
        cap.release()

    def stop(self):
        # Останавливает поток и ждёт его завершения
        self.mutex.lock()
        self._run_flag = False
        self.mutex.unlock()
        self.wait()


# Главное окно приложения
class FaceRecognizerApp(QMainWindow):

    def __init__(self):
        super().__init__()

        # Загружаем интерфейс из .ui файла
        loader = QUiLoader()
        ui_file = QFile("face_recognizer.ui")
        self.ui = loader.load(ui_file)
        ui_file.close()

        # Настраиваем главное окно
        self.setCentralWidget(self.ui.centralwidget)
        self.setWindowTitle(self.ui.windowTitle())
        self.setGeometry(self.ui.geometry())

        # Переменные для управления распознаванием и потоками
        self.video_thread = None
        self.is_recognizing = False

        # Инициализируем распознаватель лиц LBPH и каскад Хаара
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Менеджер БД
        self.db_manager = DatabaseManager()

        # Пути для хранения датасета, обученной модели и файла с пользователями
        self.dataset_dir = "dataset"
        self.trainer_dir = "trainer"
        self.users_file = "users.json"

        # Создаём папки, если их нет
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.trainer_dir, exist_ok=True)

        # Загружаем словарь пользователей {id: имя}
        self.users = self.load_users()

        # Инициализация звуковой подсистемы pygame
        pygame.mixer.init()

        # Словари для ограничения частоты приветствий и записи в БД (по таймаутам)
        self.last_greeting_time = {}   # {user_name: timestamp}
        self.last_db_log_time = {}     # {user_name: timestamp}

        # Настройка элементов интерфейса
        self.setup_ui()

        # Обновляем список пользователей в виджете
        self.update_users_list()

    def setup_ui(self):
        # Привязываем кнопки и действия к слотам
        self.ui.btnStart.clicked.connect(self.start_recognition)
        self.ui.btnStop.clicked.connect(self.stop_recognition)
        self.ui.btnAddPerson.clicked.connect(self.add_person)
        self.ui.btnDeleteUser.clicked.connect(self.delete_user)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.usersList.itemSelectionChanged.connect(self.on_user_selected)

        # Добавляем кнопку просмотра логов (её нет в .ui, создаём программно)
        self.add_logs_button()

        # Таймер для обновления статуса (каждую секунду)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)

    def add_logs_button(self):
        # Добавляет кнопку "Просмотреть логи" в layout управления
        from PySide6.QtWidgets import QPushButton

        self.btnViewLogs = QPushButton("Просмотреть логи")
        self.btnViewLogs.clicked.connect(self.view_logs)
        layout = self.ui.verticalLayout_controls
        layout.insertWidget(layout.count() - 1, self.btnViewLogs)  # вставляем перед btnDeleteUser

    def load_users(self):
        # Загружает словарь пользователей из JSON-файла
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_users(self):
        # Сохраняет словарь пользователей в JSON-файл
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)

    def update_users_list(self):
        # Обновляет QListWidget с именами и ID пользователей
        self.ui.usersList.clear()
        for user_id, user_name in self.users.items():
            item = QListWidgetItem(f"{user_id}: {user_name}")
            self.ui.usersList.addItem(item)

    def log_to_database(self, user_name, confidence):
        # Записывает событие распознавания в БД (запускается в отдельном потоке)
        try:
            success = self.db_manager.log_recognition(user_name, confidence)

            if success:
                print(f"[БД] Пользователь '{user_name}' распознан (уверенность: {confidence:.1f}%)")
                # Временно отображаем сообщение в статусной строке
                self.ui.statusLabel.setText(f"Записано в БД: {user_name}")
                original_text = self.ui.statusLabel.text()
                QTimer.singleShot(2000, lambda: self.ui.statusLabel.setText(original_text))
            else:
                print(f"[БД] Не удалось записать распознавание пользователя '{user_name}'")

        except Exception as e:
            print(f"Ошибка при записи в БД: {e}")

    def say_welcome(self, user_name):
        # Синтезирует и воспроизводит голосовое приветствие для пользователя
        try:
            greeting = f"Добро пожаловать, {user_name}"
            tts = gTTS(text=greeting, lang='ru', slow=False)

            # Создаём временный MP3-файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)

            # Воспроизводим через pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            pygame.mixer.music.unload()
            os.unlink(temp_file)  # удаляем временный файл
        except Exception as e:
            print(f"Ошибка озвучивания: {e}")

    def add_person(self):
        # Добавляет нового пользователя: запрашивает имя, захватывает 50 фото, обучает модель
        was_recognizing = self.is_recognizing
        if was_recognizing:
            self.stop_recognition()  # останавливаем распознавание на время добавления

        name, ok = QInputDialog.getText(self, "Добавление пользователя",
                                        "Введите имя пользователя:")
        if not ok or not name.strip():
            if was_recognizing:
                self.start_recognition()
            return

        name = name.strip()

        # Генерируем новый ID (минимальный свободный)
        user_id = 1
        while str(user_id) in self.users:
            user_id += 1
        user_id_str = str(user_id)

        # Создаём папку для изображений пользователя
        user_dir = os.path.join(self.dataset_dir, user_id_str)
        os.makedirs(user_dir, exist_ok=True)

        self.ui.statusLabel.setText(f"Захват изображений для {name}...")
        self.ui.statusLabel.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()

        cap = cv2.VideoCapture(0)
        count = 0
        max_samples = 50

        # Захватываем 50 кадров с лицом
        while count < max_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200))

                filename = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(filename, face_resized)

                # Рисуем рамку и прогресс на кадре
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Progress: {count}/{max_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"User: {name}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Отображаем кадр в UI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.ui.videoLabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.ui.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            QApplication.processEvents()

        cap.release()

        # Сохраняем пользователя в JSON и переобучаем модель
        self.users[user_id_str] = name
        self.save_users()
        self.train_recognizer()
        self.update_users_list()

        self.ui.statusLabel.setText(f"Пользователь {name} успешно добавлен!")
        self.ui.statusLabel.setStyleSheet("color: green; font-weight: bold;")

        if was_recognizing:
            self.start_recognition()

    def train_recognizer(self):
        # Обучает LBPH-распознаватель на всех собранных изображениях
        self.ui.statusLabel.setText("Тренировка модели...")
        self.ui.statusLabel.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()

        faces = []
        labels = []

        # Проходим по всем папкам пользователей в dataset
        for user_id in os.listdir(self.dataset_dir):
            user_dir = os.path.join(self.dataset_dir, user_id)
            if not os.path.isdir(user_dir):
                continue

            for filename in os.listdir(user_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(user_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(int(user_id))

        if faces:
            # Создаём и обучаем распознаватель с заданными параметрами
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=80.0
            )
            self.face_recognizer.train(faces, np.array(labels))

            # Сохраняем модель в файл
            trainer_path = os.path.join(self.trainer_dir, "trainer.yml")
            self.face_recognizer.save(trainer_path)

            self.ui.statusLabel.setText(f"Модель обучена на {len(faces)} изображениях")
            self.ui.statusLabel.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.ui.statusLabel.setText("Нет данных для обучения модели")
            self.ui.statusLabel.setStyleSheet("color: red;")

    def load_recognizer(self):
        # Загружает сохранённую модель из файла trainer.yml
        trainer_path = os.path.join(self.trainer_dir, "trainer.yml")
        if os.path.exists(trainer_path):
            self.face_recognizer.read(trainer_path)
            return True
        return False

    def start_recognition(self):
        # Запускает видеопоток и распознавание лиц
        if not self.load_recognizer():
            QMessageBox.warning(self, "Предупреждение",
                                "Сначала добавьте хотя бы одного пользователя!")
            return

        if self.is_recognizing:
            return
        self.is_recognizing = True
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.video_thread.start()
        self.ui.statusLabel.setText("Распознавание запущено")
        self.ui.statusLabel.setStyleSheet("color: green; font-weight: bold;")

    def stop_recognition(self):
        # Останавливает видеопоток и распознавание
        self.is_recognizing = False

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        # Очищаем виджет видео
        self.ui.videoLabel.setText("Видео остановлено")
        self.ui.videoLabel.setStyleSheet("background-color: black; color: white;")
        self.ui.videoLabel.setAlignment(Qt.AlignCenter)
        self.ui.statusLabel.setText("Распознавание остановлено")
        self.ui.statusLabel.setStyleSheet("color: red; font-weight: bold;")

    def update_video_frame(self, frame):
        # Обрабатывает каждый кадр: детектирует лица и выполняет распознавание
        if not self.is_recognizing:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))

            try:
                label, confidence = self.face_recognizer.predict(face_resized)

                # Если уверенность выше порога (меньше число — лучше)
                if confidence < 80:
                    user_name = self.users.get(str(label), f"User_{label}")
                    text = f"{user_name} ({confidence:.1f}%)"
                    color = (0, 255, 0)

                    current_time = time.time()
                    last_time = self.last_greeting_time.get(user_name, 0)

                    # Приветствие не чаще 1 раза в 30 секунд
                    if current_time - last_time > 30:
                        self.last_greeting_time[user_name] = current_time
                        threading.Thread(target=self.say_welcome, args=(user_name,), daemon=True).start()

                    # Запись в БД не чаще 1 раза в 60 секунд
                    last_db_time = self.last_db_log_time.get(user_name, 0)
                    if current_time - last_db_time > 60:
                        self.last_db_log_time[user_name] = current_time
                        threading.Thread(target=self.log_to_database,
                                         args=(user_name, confidence), daemon=True).start()
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
            except:
                text = "Unknown"
                color = (0, 0, 255)

            # Рисуем прямоугольник и подпись на кадре
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Конвертируем кадр в формат QImage и отображаем в виджете
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.videoLabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.ui.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def view_logs(self):
        # Показывает диалог с последними записями из БД
        logs = self.db_manager.get_recent_logs(50)

        if not logs:
            QMessageBox.information(self, "Логи распознавания",
                                    "Нет записей в базе данных")
            return

        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("Логи распознавания из БД")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        log_text = "ПОСЛЕДНИЕ СОБЫТИЯ РАСПОЗНАВАНИЯ:\n"
        log_text += "=" * 50 + "\n\n"

        for log in logs:
            log_text += f"Пользователь: {log['username']}\n"
            log_text += f"Уверенность: {log['confidence']:.1f}%\n"
            log_text += f"Время: {log['recognition_time']}\n"
            log_text += "-" * 40 + "\n"

        text_edit.setText(log_text)
        layout.addWidget(text_edit)
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def delete_user(self):
        # Удаляет выбранного пользователя: стирает его фото и удаляет из списка
        current_item = self.ui.usersList.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Предупреждение", "Выберите пользователя для удаления!")
            return

        user_text = current_item.text()
        user_id = user_text.split(":")[0]
        user_name = self.users.get(user_id, "Unknown")

        reply = QMessageBox.question(self, "Подтверждение",
                                     f"Вы уверены, что хотите удалить пользователя {user_name}?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            was_recognizing = self.is_recognizing
            if was_recognizing:
                self.stop_recognition()

            # Удаляем папку с изображениями
            user_dir = os.path.join(self.dataset_dir, user_id)
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)

            # Удаляем из словаря и сохраняем
            del self.users[user_id]
            self.save_users()

            # Переобучаем модель, если остались другие пользователи
            if self.users:
                self.train_recognizer()
            else:
                trainer_path = os.path.join(self.trainer_dir, "trainer.yml")
                if os.path.exists(trainer_path):
                    os.remove(trainer_path)
                self.ui.statusLabel.setText("Нет пользователей в базе")
                self.ui.statusLabel.setStyleSheet("color: orange;")

            self.update_users_list()

            if was_recognizing and self.users:
                self.start_recognition()

            self.ui.statusLabel.setText(f"Пользователь {user_name} удален")

    def on_user_selected(self):
        # Слот для события выбора пользователя в списке (пока ничего не делает)
        pass

    def update_status(self):
        # Обновляет текст статуса, если распознавание активно
        if self.is_recognizing:
            self.ui.statusLabel.setText("Распознавание активно")
            self.ui.statusLabel.setStyleSheet("color: green; font-weight: bold;")

    def closeEvent(self, event):
        # При закрытии окна останавливаем распознавание и закрываем соединение с БД
        self.stop_recognition()
        self.db_manager.close()
        event.accept()


def main():
    # Точка входа в приложение
    app = QApplication(sys.argv)
    window = FaceRecognizerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()