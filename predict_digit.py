import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2  # Still needed for preprocessing
import numpy as np
import tensorflow as tf
import os

# --- Global model variable ---
MODEL = None
MODEL_PATH = "mnist_model.h5"  # Ensure this path is correct


def load_keras_model():
    """Loads the Keras model from MODEL_PATH."""
    global MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        # This error will be shown in GUI status as well
        return False
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        # Optional: Compile if needed, though usually not for H5 predict if optimizer state is saved.
        # MODEL.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None  # Ensure model is None if loading fails
        return False


# --- Image Preprocessing Function ---
def preprocess_image(image_path, output_size=28):
    """
    Loads an image, preprocesses it for MNIST model prediction.
    Converts to white digit on black background, resizes to 28x28, centers digit.
    Returns:
        - img_reshaped_for_model: Numpy array for model input (1, 28, 28, 1).
        - img_canvas: Numpy array (28, 28, uint8) for display.
    """
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to load image from path: {image_path}")

        # 1. Invert colors: MNIST expects white digits on a black background.
        # Assumes input is typically black digit on white background.
        img_inverted = cv2.bitwise_not(img_gray)

        # 2. Binarization using Otsu's method.
        # Results in a white digit (255) on a black background (0).
        _, img_thresh = cv2.threshold(
            img_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 3. Find contours to locate the digit.
        # Use .copy() as findContours can modify the source image.
        contours, _ = cv2.findContours(
            img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        img_canvas = np.zeros(
            (output_size, output_size), dtype=np.uint8
        )  # Final 28x28 canvas

        # Minimum contour area to be considered a digit.
        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 10:
            # If no significant contours, treat as a blank image.
            print(
                f"Warning: No significant contours found in {image_path}. Processing as blank image."
            )
            # img_canvas remains black
        else:
            contour = max(
                contours, key=cv2.contourArea
            )  # Largest contour is assumed to be the digit
            x, y, w, h = cv2.boundingRect(contour)
            img_digit_cropped = img_thresh[y : y + h, x : x + w]

            # 4. Scale and center the digit onto the canvas.
            # MNIST digits are typically ~20x20px centered in a 28x28px image.
            margin = (
                4  # Margin on each side (e.g., 4 pixels results in a 20x20 target area)
            )
            target_dim = output_size - 2 * margin

            current_h_cropped, current_w_cropped = img_digit_cropped.shape

            # Scale digit, preserving aspect ratio, to fit into target_dim x target_dim.
            if current_w_cropped > current_h_cropped:
                new_w = target_dim
                new_h = (
                    int(current_h_cropped * new_w / current_w_cropped)
                    if current_w_cropped > 0
                    else 0
                )
            else:
                new_h = target_dim
                new_w = (
                    int(current_w_cropped * new_h / current_h_cropped)
                    if current_h_cropped > 0
                    else 0
                )

            new_w = max(1, new_w)
            new_h = max(1, new_h)  # Ensure dimensions are at least 1

            img_digit_resized = cv2.resize(
                img_digit_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            # Calculate paste position for centering
            paste_x = (output_size - new_w) // 2
            paste_y = (output_size - new_h) // 2

            img_canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = (
                img_digit_resized
            )

        # Optional: Morphological operations (e.g., to remove small noise if needed)
        # kernel_morph = np.ones((1, 1), np.uint8) # Smaller kernel might be safer
        # img_canvas = cv2.morphologyEx(img_canvas, cv2.MORPH_OPEN, kernel_morph)

        # Normalize pixel values to [0, 1] for the model.
        img_normalized = img_canvas.astype("float32") / 255.0

        # Reshape for the model: (1, height, width, channels).
        img_reshaped_for_model = img_normalized.reshape(1, output_size, output_size, 1)

        return img_reshaped_for_model, img_canvas

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None, None  # Indicate failure


# --- Prediction Function ---
def get_prediction_from_model(processed_image_for_model):
    """Predicts the digit from the preprocessed image using the global MODEL."""
    if MODEL is None:
        print("Error: Model not loaded.")
        return None, 0, []  # Predicted digit, confidence, probabilities array
    try:
        # model.predict returns a list of arrays, take the first element for single input
        prediction_array = MODEL.predict(processed_image_for_model, verbose=0)
        prediction_probabilities = prediction_array[0]

        predicted_digit = np.argmax(prediction_probabilities)
        confidence = np.max(prediction_probabilities) * 100
        return predicted_digit, confidence, prediction_probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0, []


# --- GUI Application ---
class DigitRecognizerApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("500x650")  # Adjusted size for probability text

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Provides a modern look

        # Main frame
        main_frame = ttk.Frame(root_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control Frame (Button and Status)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10, fill=tk.X)

        self.load_button = ttk.Button(
            control_frame, text="Load Image", command=self.run_prediction_pipeline
        )
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = ttk.Label(
            control_frame, text="Initializing...", anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Image Display Frame
        image_display_frame = ttk.LabelFrame(
            main_frame, text="Preprocessed Image (28x28)", padding="10"
        )
        image_display_frame.pack(pady=10, fill=tk.X)

        self.image_label = ttk.Label(image_display_frame)  # Will hold the image
        self.image_label.pack(pady=10, anchor=tk.CENTER)
        self.display_blank_preview_image(
            280, 280
        )  # Display a blank image initially (scaled up)

        # Prediction Display Frame
        prediction_frame = ttk.LabelFrame(
            main_frame, text="Prediction Result", padding="10"
        )
        prediction_frame.pack(pady=10, fill=tk.X)

        self.prediction_text_var = tk.StringVar()
        self.prediction_label = ttk.Label(
            prediction_frame,
            textvariable=self.prediction_text_var,
            font=("Helvetica", 18, "bold"),
            anchor=tk.CENTER,
        )
        self.prediction_label.pack(fill=tk.X)
        self.prediction_text_var.set("Predicted Digit: -")

        # Probabilities Display Frame
        probabilities_frame = ttk.LabelFrame(
            main_frame, text="Prediction Probabilities", padding="10"
        )
        probabilities_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.probabilities_text_widget = tk.Text(
            probabilities_frame,
            height=11,
            width=40,
            wrap=tk.WORD,
            relief=tk.SOLID,
            borderwidth=1,
            font=("Courier New", 10),
        )
        self.probabilities_text_widget.pack(fill=tk.BOTH, expand=True)
        self.probabilities_text_widget.insert(
            tk.END, "Load an image to see probabilities."
        )
        self.probabilities_text_widget.config(state=tk.DISABLED)  # Make read-only

        # Attempt to load model on startup
        if load_keras_model():
            self.status_label.config(text="Model loaded. Ready to predict.")
        else:
            self.status_label.config(
                text=f"Error: Model '{MODEL_PATH}' not found or corrupt. Check console."
            )
            self.load_button.config(
                state=tk.DISABLED
            )  # Disable button if model fails to load

    def display_blank_preview_image(self, display_width, display_height):
        # Create a small 28x28 light gray PIL Image
        blank_pil_img = Image.new("L", (28, 28), color="lightgray")
        # Resize for display using NEAREST to keep pixels sharp
        blank_pil_img_resized = blank_pil_img.resize(
            (display_width, display_height), Image.NEAREST
        )

        self.current_img_tk = ImageTk.PhotoImage(blank_pil_img_resized)
        self.image_label.config(image=self.current_img_tk)
        # Keep a reference to prevent garbage collection
        self.image_label.image = self.current_img_tk

    def run_prediction_pipeline(self):
        if MODEL is None:  # Should be prevented by button state, but double-check
            self.status_label.config(text="Error: Model is not loaded.")
            return

        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*"),
            ),
        )
        if not file_path:  # User cancelled the dialog
            return

        self.status_label.config(text=f"Processing: {os.path.basename(file_path)}...")
        self.root.update_idletasks()  # Force GUI update

        # Preprocess the image
        processed_model_input, displayable_canvas = preprocess_image(file_path)

        if processed_model_input is None or displayable_canvas is None:
            self.prediction_text_var.set("Error: Image processing failed.")
            self.status_label.config(text="Image processing failed. Try another image.")
            self.probabilities_text_widget.config(state=tk.NORMAL)
            self.probabilities_text_widget.delete(1.0, tk.END)
            self.probabilities_text_widget.insert(tk.END, "Image processing failed.")
            self.probabilities_text_widget.config(state=tk.DISABLED)
            self.display_blank_preview_image(280, 280)  # Reset image display
            return

        # Display the preprocessed image (displayable_canvas is a 28x28 numpy array)
        pil_img = Image.fromarray(displayable_canvas).convert("L")
        img_display_size = 280  # Desired display size (e.g., 28 * 10 for 10x zoom)
        pil_img_resized = pil_img.resize(
            (img_display_size, img_display_size), Image.NEAREST
        )

        self.current_img_tk = ImageTk.PhotoImage(pil_img_resized)
        self.image_label.config(image=self.current_img_tk)
        self.image_label.image = self.current_img_tk  # Keep a reference!

        # Get prediction
        predicted_digit, confidence, probabilities = get_prediction_from_model(
            processed_model_input
        )

        if predicted_digit is not None:
            self.prediction_text_var.set(
                f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)"
            )
            self.status_label.config(text="Prediction complete.")

            self.probabilities_text_widget.config(state=tk.NORMAL)  # Enable editing
            self.probabilities_text_widget.delete(1.0, tk.END)  # Clear previous content
            prob_str = ""
            for i, prob_val in enumerate(probabilities):
                prob_str += (
                    f"Digit {i}: {prob_val*100:>6.2f}%\n"  # Format for alignment
                )
            self.probabilities_text_widget.insert(tk.END, prob_str)
            self.probabilities_text_widget.config(
                state=tk.DISABLED
            )  # Make read-only again
        else:
            self.prediction_text_var.set("Prediction failed.")
            self.status_label.config(
                text="Prediction failed. Check console for errors."
            )
            self.probabilities_text_widget.config(state=tk.NORMAL)
            self.probabilities_text_widget.delete(1.0, tk.END)
            self.probabilities_text_widget.insert(tk.END, "Prediction failed.")
            self.probabilities_text_widget.config(state=tk.DISABLED)


if __name__ == "__main__":
    gui_root = tk.Tk()
    app = DigitRecognizerApp(gui_root)
    gui_root.mainloop()
