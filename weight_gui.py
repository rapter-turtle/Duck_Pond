import sys
import rospy
from PyQt5 import QtWidgets, QtCore, QtGui
from std_msgs.msg import Float64MultiArray

class WeightPublisher(QtCore.QObject):
    def __init__(self):
        super().__init__()
        # ROS Publisher for the weight matrix
        self.weight_pub = rospy.Publisher('/weight', Float64MultiArray, queue_size=10)
        self.q_weights = [1, 1, 20, 30, 10, 5, 0.0001, 0.0001]
        self.r_weights = [0.001, 0.001]

        # Timer for publishing weights at 10 Hz
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_weights)
        self.timer.start(100)  # 100 ms interval for 10 Hz

    def publish_weights(self):
        # Prepare and publish the weights
        weights = Float64MultiArray()
        weights.data = self.q_weights + self.r_weights  # Combine Q and R weights into a single list
        self.weight_pub.publish(weights)
        rospy.loginfo(f"Published weights: {weights.data}")

    def set_weights(self, q_weights, r_weights):
        self.q_weights = q_weights
        self.r_weights = r_weights

class WeightGui(QtWidgets.QWidget):
    def __init__(self, weight_publisher):
        super().__init__()

        self.weight_publisher = weight_publisher

        # Set up the GUI layout
        self.setWindowTitle('MPC Weight Tuning')
        self.setGeometry(100, 100, 600, 400)

        layout = QtWidgets.QVBoxLayout()

        # Instructions Label
        # self.instruction_label = QtWidgets.QLabel(
        #     "Enter each weight coefficient separately."
        # )
        # layout.addWidget(self.instruction_label)

        # Weight labels
        weight_labels = [
            'x', 'y', 'psi', 'u', 'v', 'r', 
            'thrust_left', 'thrust_right', 
            'thrust_dot_left', 'thrust_dot_right'
        ]

        # Set font for labels
        font = QtGui.QFont()
        font.setPointSize(10)  # Set base font size for smaller labels

        # Q Weight Inputs
        self.q_weights_inputs = []
        self.q_weights_label = QtWidgets.QLabel('Q Weights:')
        self.q_weights_label.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))  # Increase font size
        layout.addWidget(self.q_weights_label)
        q_weights_layout = QtWidgets.QFormLayout()
        for i in range(8):
            label = QtWidgets.QLabel(weight_labels[i])
            label.setFont(font)
            input_field = QtWidgets.QLineEdit(str(self.weight_publisher.q_weights[i]))
            input_field.setFixedWidth(50)
            q_weights_layout.addRow(label, input_field)
            self.q_weights_inputs.append(input_field)
        layout.addLayout(q_weights_layout)

        # R Weight Inputs
        self.r_weights_inputs = []
        self.r_weights_label = QtWidgets.QLabel('R Weights:')
        self.r_weights_label.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))  # Increase font size
        layout.addWidget(self.r_weights_label)
        r_weights_layout = QtWidgets.QFormLayout()
        for i in range(2):
            label = QtWidgets.QLabel(weight_labels[8 + i])
            label.setFont(font)
            input_field = QtWidgets.QLineEdit(str(self.weight_publisher.r_weights[i]))
            input_field.setFixedWidth(50)
            r_weights_layout.addRow(label, input_field)
            self.r_weights_inputs.append(input_field)
        layout.addLayout(r_weights_layout)

        # Update Button
        self.update_button = QtWidgets.QPushButton('Update Weights')
        self.update_button.setFont(QtGui.QFont('Arial', 10))  # Set font size for button
        self.update_button.clicked.connect(self.update_weights)
        layout.addWidget(self.update_button)

        # Set the layout
        self.setLayout(layout)

    def update_weights(self):
        try:
            # Parse input weights
            q_weights = [float(field.text()) for field in self.q_weights_inputs]
            r_weights = [float(field.text()) for field in self.r_weights_inputs]

            # Set new weights in the publisher
            self.weight_publisher.set_weights(q_weights, r_weights)
        except ValueError:
            # Show error message if parsing fails
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter valid numbers for weights.")

def main():
    # Initialize the ROS node
    rospy.init_node('weight_publisher_node', anonymous=True)

    # Create the PyQt application
    app = QtWidgets.QApplication(sys.argv)

    # Create and show the GUI
    weight_publisher = WeightPublisher()
    gui = WeightGui(weight_publisher)
    gui.show()

    # Run the Qt application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
