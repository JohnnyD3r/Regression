import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton, QVBoxLayout, QMessageBox, QDialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

class HousePP(QWidget):
    def __init__(self):
        super().__init__()

        self.init_userinterface()

        self.TrainingModel()

    def init_userinterface(self):
        self.setWindowTitle("House Price Predictor - Statistics/Predictions")
        self.setGeometry(300, 300, 400, 300) # (x Cordinates, y Cordinates, Width, Height)

        self.bedrooms_l = QLabel("Give Number Of Bedrooms:") # Asking For Number Of Bedrooms
        self.bedrooms_i = QLineEdit(self) # Input

        self.bathrooms_l = QLabel("Give Number Of Bathrooms:") # Asking For Number Of Bathrooms
        self.bathrooms_i = QLineEdit(self) # Input

        self.ac_l = QLabel("Air Condtioning:") # Asking For Air Condtioning Status
        self.ac_i = QComboBox(self) # Initializing Dropdown 
        self.ac_i.addItems(["Yes", "No"]) # Adding Dropdown Items

        self.parkspace_l = QLabel("Give Number Of Parking Spaces:") # Asking For Number Of Parking Spaces
        self.parkspace_i = QLineEdit(self) # Input

        self.furnishes_l = QLabel("Furnishing Status:")
        self.furnishes_i = QComboBox(self)
        self.furnishes_i.addItems(["Furnished", "Semi-Furnished", "Unfurnished"])

        self.submit = QPushButton("Submit", self) # Submit
        self.submit.clicked.connect(self.Statistics)

        self.predict = QPushButton("Predict", self) # Price Prediction 
        self.predict.clicked.connect(self.pp)

        self.regression = QPushButton("Show Regression Plot", self) # Show Regression Plot
        self.regression.clicked.connect(self.plot_regression)

        layout = QVBoxLayout()
        layout.addWidget(self.bedrooms_l)
        layout.addWidget(self.bedrooms_i)
        layout.addWidget(self.bathrooms_l)
        layout.addWidget(self.bathrooms_i)
        layout.addWidget(self.ac_l)
        layout.addWidget(self.ac_i)
        layout.addWidget(self.parkspace_l)
        layout.addWidget(self.parkspace_i)
        layout.addWidget(self.furnishes_l)
        layout.addWidget(self.furnishes_i)
        layout.addWidget(self.submit)
        layout.addWidget(self.predict)
        layout.addWidget(self.regression)

        self.setLayout(layout)

    def TrainingModel(self):
        self.data = pd.read_csv("C:/Users/giann/Downloads/Housing.csv")
        
        x = pd.DataFrame({'area': self.data['area']})  
        y = self.data['price']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(x_train, y_train)

        self.model = model

    def Statistics(self):
        try: 
            bedrooms = int(self.bedrooms_i.text())
            bathrooms = int(self.bathrooms_i.text())
            ac = self.ac_i.currentText().lower()
            parkspace = int(self.parkspace_i.text())
            furnishes = self.furnishes_i.currentText().lower()

            s_data = self.data[
                (self.data['bedrooms'] == bedrooms) &
                (self.data['bathrooms'] == bathrooms) &
                (self.data['airconditioning'] == ac) &
                (self.data['parking'] == parkspace) &
                (self.data['furnishingstatus'] == furnishes) 
            ]

            if not s_data.empty:
                avr_p = s_data['price'].mean()

                if len(s_data) > 1:
                    price_std = s_data['price'].std()
                else:
                    price_std = 0
                
                price_med = s_data["price"].median()

                message = f"\nStatistical Information: \n Average Price: {avr_p:,.2f}$ \n Price Deviation: {price_std:,.2f}$ \n Median Price: {price_med:,.2f}$ \n"

                QMessageBox.information(self, "Statistics", message)

            else:

                QMessageBox.warning(self, "No Data", "No Matching Data Found!")
        
        except ValueError:

            QMessageBox.warning(self, "Invalid Input", "Please Enter Valid Numerical Values For Bedrooms, Bathrooms And Parking!")

    def plot_regression(self):
            
            plt.figure(figsize=(10,6))
            plt.scatter(self.data['area'], self.data['price'], label='Data Points')
            plt.plot(self.data['area'], self.model.predict(self.data[['area']]), color='red', label='Regression Line')
            plt.title('Linear Regression Plot')
            plt.xlabel('Area (sqft)')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            plt.show()


    def pp(self): # Predict Price

        predict_window = QDialog(self)
        predict_window.setWindowTitle("Price Prediction And Property Category")

        area_l = QLabel("Area:")
        area_i = QLineEdit(predict_window)
        predict_b = QPushButton('Predict', predict_window)
        predict_b.clicked.connect(lambda: self.print_pp(area_i.text(), predict_window))

        layout = QVBoxLayout()
        layout.addWidget(area_l)
        layout.addWidget(area_i)
        layout.addWidget(predict_b)
        predict_window.setLayout(layout)

        predict_window.exec()

    def print_pp (self, area, predict_window): # Printi Of Price Prediction

        try:

            area = float(area)
            features = np.array([area]).reshape(1, -1)
            p_price = self.model.predict(features)[0] # p_price = Predicted Price

            if p_price > self.data['price'].max():
                property_cat = 'Unreachable'
            elif p_price > self.data['price'].mean():
                property_cat = 'Intermediate'
            else:
                property_cat = 'Surprisingly Cheap'

            QMessageBox.information(predict_window, "Predicted Price And Property Category", f'Predicted Price For {area} (sqft) area: {p_price:,.2f}$ \n Property Category: {property_cat}')

            predict_window.accept()
        
        except ValueError:

            QMessageBox.warning(predict_window, "Invalid Input", "Please Enter A Valid Numerical Value For Area!")

if __name__ == '__main__':
    application = QApplication(sys.argv)
    window = HousePP()
    window.show()
    sys.exit(application.exec_())