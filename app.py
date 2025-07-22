import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from PIL import Image, ImageTk

# Your feature extraction functions go here (remove_green_pixels, extract_features, etc.)
# ... (Paste all your feature extraction code here, unchanged)

labels = ['Vitamin A', 'Vitamin B', 'Vitamin C','Vitamin D','Vitamin E']

class VitaminDeficiencyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vitamin Deficiency Detector")
        self.geometry("900x600")

        self.dataset_path = ""
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.svm_model = None
        self.cnn_model = None

        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        # Dataset selection
        btn_load = tk.Button(self, text="Select Dataset Folder", command=self.select_dataset)
        btn_load.pack(pady=5)

        # Feature extraction button
        btn_extract = tk.Button(self, text="Extract Features & Prepare Data", command=self.extract_features_and_prepare)
        btn_extract.pack(pady=5)

        # Train and evaluate SVM
        btn_svm = tk.Button(self, text="Run SVM", command=self.run_svm)
        btn_svm.pack(pady=5)

        # Train and evaluate CNN
        btn_cnn = tk.Button(self, text="Run CNN", command=self.run_cnn)
        btn_cnn.pack(pady=5)

        # Predict single image
        btn_predict = tk.Button(self, text="Predict Single Image", command=self.predict_image)
        btn_predict.pack(pady=5)

        # Text box for logs/output
        self.log_text = scrolledtext.ScrolledText(self, width=100, height=20)
        self.log_text.pack(pady=10)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.update()

    def select_dataset(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path = folder
            self.log(f"Dataset folder selected: {folder}")

    def extract_features_and_prepare(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return

        self.log("Extracting features from dataset...")

        X = []
        Y = []

        for root, dirs, files in os.walk(self.dataset_path):
            class_name = os.path.basename(root)
            if class_name not in labels:
                continue
            class_id = labels.index(class_name)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (64,64))
                        feats = extract_features(img)
                        X.append(feats)
                        Y.append(class_id)
                        self.log(f"Processed {file} in class {class_name}")
                    except Exception as e:
                        self.log(f"Error processing {file}: {str(e)}")

        if len(X) == 0:
            messagebox.showerror("Error", "No images found or invalid dataset structure.")
            return

        self.X = np.array(X, dtype=np.float32) / 255.0
        self.Y = np.array(Y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        self.log(f"Feature extraction complete. Total samples: {len(self.X)}")
        self.log(f"Training samples: {len(self.X_train)}, Testing samples: {len(self.X_test)}")

    def run_svm(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please extract features and prepare data first.")
            return

        self.log("Training SVM model...")
        self.svm_model = svm.SVC()
        self.svm_model.fit(self.X_train, self.y_train)

        self.log("Predicting test data with SVM...")
        preds = self.svm_model.predict(self.X_test)

        self.display_metrics("SVM", preds, self.y_test)

    def run_cnn(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Error", "Please extract features and prepare data first.")
            return

        self.log("Preparing data for CNN...")
        Y_cat = to_categorical(self.Y, num_classes=len(labels))
        XX = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1, 1))
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(XX, Y_cat, test_size=0.2, random_state=42)

        self.log("Building CNN model...")
        model = Sequential()
        model.add(Convolution2D(32, (1, 1), input_shape=(XX.shape[1], XX.shape[2], XX.shape[3]), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Convolution2D(32, (1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.log("Training CNN model...")
        model.fit(XX, Y_cat, batch_size=12, epochs=10, verbose=2)

        self.cnn_model = model

        self.log("Predicting test data with CNN...")
        preds = model.predict(X_test_cnn)
        preds = np.argmax(preds, axis=1)
        y_test_labels = np.argmax(y_test_cnn, axis=1)

        self.display_metrics("CNN", preds, y_test_labels)

    def display_metrics(self, model_name, preds, y_true):
        acc = accuracy_score(y_true, preds) * 100
        prec = precision_score(y_true, preds, average='macro') * 100
        rec = recall_score(y_true, preds, average='macro') * 100
        f1 = f1_score(y_true, preds, average='macro') * 100

        self.log(f"{model_name} Accuracy: {acc:.2f}%")
        self.log(f"{model_name} Precision: {prec:.2f}%")
        self.log(f"{model_name} Recall: {rec:.2f}%")
        self.log(f"{model_name} F1 Score: {f1:.2f}%")

        # Confusion matrix plot
        cm = confusion_matrix(y_true, preds)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels, cmap="viridis")
        plt.title(f"{model_name} Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def predict_image(self):
        if self.cnn_model is None:
            messagebox.showerror("Error", "Please train the CNN model first.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, (64, 64))
        feats = extract_features(img_resized)
        test_feat = np.array([feats], dtype=np.float32) / 255.0
        test_feat = np.reshape(test_feat, (1, test_feat.shape[1], 1, 1))

        pred = self.cnn_model.predict(test_feat)
        pred_class = np.argmax(pred)
        pred_label = labels[pred_class]

        self.log(f"Prediction: {pred_label}")

        # Show image with label
        img_display = cv2.resize(img, (800, 400))
        cv2.putText(img_display, f'Vitamin Deficiency Predicted: {pred_label}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Prediction", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VitaminDeficiencyApp()
    app.mainloop()
