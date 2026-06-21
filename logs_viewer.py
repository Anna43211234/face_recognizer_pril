import sys
import csv
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                               QTableWidgetItem, QHeaderView)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QTimer
from PySide6.QtGui import QColor
import mysql.connector
from mysql.connector import Error


class LogsViewerApp(QMainWindow):

    def __init__(self):
        super().__init__()

        loader = QUiLoader()
        ui_file = QFile("logs_viewer.ui")
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.setCentralWidget(self.ui.centralwidget)
        self.setWindowTitle(self.ui.windowTitle())
        self.setGeometry(self.ui.geometry())

        self.db_connection = None
        self.connect_to_database()

        self.setup_table()
        self.load_users_for_filter()
        self.setup_signals()

        self.load_logs()

        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self.load_logs)
        self.auto_refresh_timer.start(3000)  # 3 секунды

        self.ui.statusbar.showMessage("Автообновление каждые 3 секунды", 3000)

    def connect_to_database(self):
        try:
            self.db_connection = mysql.connector.connect(
                host='MySQL-8.4',
                database='face_recognition_db',
                user='root',
                password=''
            )
            self.ui.statusbar.showMessage("✓ Подключено к базе данных", 3000)
        except Error as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться к БД:\n{e}")
            self.ui.statusbar.showMessage("✗ Ошибка подключения к БД", 5000)

    def setup_table(self):
        table = self.ui.tableLogs
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Пользователь
        header.setSectionResizeMode(2, QHeaderView.Stretch)          # Сообщение
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) # Дата
        table.setAlternatingRowColors(True)

    def setup_signals(self):
        self.ui.comboUserFilter.currentIndexChanged.connect(self.load_logs)
        self.ui.btnClearLogs.clicked.connect(self.clear_all_logs)
        self.ui.btnDeleteSelected.clicked.connect(self.delete_selected)
        self.ui.btnExportCSV.clicked.connect(self.export_to_csv)

    def load_users_for_filter(self):
        if not self.db_connection:
            return
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT DISTINCT username FROM recognition_logs ORDER BY username")
            users = cursor.fetchall()
            cursor.close()

            self.ui.comboUserFilter.clear()
            self.ui.comboUserFilter.addItem("Все пользователи", None)
            for user in users:
                self.ui.comboUserFilter.addItem(user[0], user[0])
        except Error as e:
            print(f"Ошибка загрузки пользователей: {e}")

    def load_logs(self):
        if not self.db_connection:
            return

        try:
            cursor = self.db_connection.cursor(dictionary=True)

            query = "SELECT * FROM recognition_logs"
            params = []

            user_filter = self.ui.comboUserFilter.currentData()
            if user_filter:
                query += " WHERE username = %s"
                params.append(user_filter)

            query += " ORDER BY recognition_time DESC"

            cursor.execute(query, params)
            logs = cursor.fetchall()
            cursor.close()

            self.display_logs(logs)
            self.update_statistics(logs)

            if not user_filter:
                self.load_users_for_filter()

            self.ui.statusbar.showMessage(f"Загружено {len(logs)} записей | Автообновление", 2000)

        except Error as e:
            print(f"Ошибка загрузки логов: {e}")

    def display_logs(self, logs):
        table = self.ui.tableLogs
        table.setRowCount(len(logs))

        for row, log in enumerate(logs):
            id_item = QTableWidgetItem(str(log['id']))
            id_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, id_item)

            user_item = QTableWidgetItem(log['username'])
            table.setItem(row, 1, user_item)

            message = f"Пользователь {log['username']} распознан"
            if log['confidence']:
                message += f" (уверенность: {log['confidence']:.1f}%)"
            message_item = QTableWidgetItem(message)
            table.setItem(row, 2, message_item)

            time_str = log['recognition_time'].strftime("%d.%m.%Y %H:%M:%S")
            time_item = QTableWidgetItem(time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 3, time_item)

            if log['confidence'] and log['confidence'] < 60:
                for col in range(4):
                    if table.item(row, col):
                        table.item(row, col).setBackground(QColor(255, 200, 200))

    def update_statistics(self, logs):
        if not logs:
            self.ui.labelUniqueUsers.setText("Уникальных пользователей: 0")
            return
        unique_users = len(set(log['username'] for log in logs))
        self.ui.labelUniqueUsers.setText(f"Уникальных пользователей: {unique_users}")

    def export_to_csv(self):
        from PySide6.QtWidgets import QFileDialog

        if not self.db_connection:
            QMessageBox.warning(self, "Ошибка", "Нет подключения к базе данных")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV файл", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            cursor = self.db_connection.cursor(dictionary=True)

            query = "SELECT * FROM recognition_logs"
            params = []

            user_filter = self.ui.comboUserFilter.currentData()
            if user_filter:
                query += " WHERE username = %s"
                params.append(user_filter)

            query += " ORDER BY recognition_time DESC"

            cursor.execute(query, params)
            logs = cursor.fetchall()
            cursor.close()

            with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ID', 'Пользователь', 'Сообщение', 'Уверенность (%)', 'Время распознавания'])

                for log in logs:
                    message = f"Пользователь {log['username']} распознан"
                    writer.writerow([
                        log['id'],
                        log['username'],
                        message,
                        f"{log['confidence']:.1f}" if log['confidence'] else "N/A",
                        log['recognition_time'].strftime("%Y-%m-%d %H:%M:%S")
                    ])

            QMessageBox.information(self, "Успех", f"Данные экспортированы в:\n{file_path}")
            self.ui.statusbar.showMessage(f"Экспортировано {len(logs)} записей", 3000)

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось экспортировать данные:\n{e}")

    def delete_selected(self):
        selected_rows = set()
        for item in self.ui.tableLogs.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.warning(self, "Предупреждение", "Выберите записи для удаления")
            return

        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Удалить {len(selected_rows)} запись(ей)?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                cursor = self.db_connection.cursor()
                deleted_count = 0
                for row in selected_rows:
                    log_id = self.ui.tableLogs.item(row, 0).text()
                    cursor.execute("DELETE FROM recognition_logs WHERE id = %s", (log_id,))
                    deleted_count += cursor.rowcount
                self.db_connection.commit()
                cursor.close()

                self.load_logs()
                self.ui.statusbar.showMessage(f"Удалено {deleted_count} записей", 3000)

                if deleted_count > 0:
                    QMessageBox.information(self, "Успех", f"Удалено {deleted_count} записей")

            except Error as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось удалить записи:\n{e}")

    def clear_all_logs(self):
        reply = QMessageBox.question(
            self, "Подтверждение",
            "Вы действительно хотите удалить ВСЕ записи из базы данных?\nЭто действие нельзя отменить!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("DELETE FROM recognition_logs")
                deleted_count = cursor.rowcount
                self.db_connection.commit()
                cursor.close()

                self.load_logs()
                self.load_users_for_filter()
                self.ui.statusbar.showMessage(f"Удалено {deleted_count} записей", 3000)
                QMessageBox.information(self, "Успех", f"Все логи очищены. Удалено {deleted_count} записей")

            except Error as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось очистить логи:\n{e}")

    def closeEvent(self, event):
        if self.auto_refresh_timer:
            self.auto_refresh_timer.stop()
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LogsViewerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()