import streamlit as st
import cv2
import torch
import numpy as np

import os
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# ================= MODEL =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "face_landmarks.pth")


    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    #detector = dlib.get_frontal_face_detector()
    return model, device

model, device = load_model()

# ================= HELPERS =================
def process_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = x, y, x + w, y + h

        crop = gray[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (224, 224))
        tensor = torch.tensor(crop).unsqueeze(0).unsqueeze(0).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        tensor = tensor.to(device)

        with torch.no_grad():
            preds = model(tensor)

        landmarks = (preds.cpu().view(-1, 2) + 0.5) * 224

        for (x, y) in landmarks:
            cv2.circle(image_np, (x1 + int(x * (x2 - x1) / 224),
                                   y1 + int(y * (y2 - y1) / 224)),
                       1, (255, 0, 0), -1)

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image_np, len(faces)

# ================= UI =================
st.title("Face Landmark Detection System")

st.markdown("### 🔐 Data Usage Consent")
consent = st.checkbox("I allow my data to be used for research / improvement purposes")

st.info(
    "If you do NOT give consent, your data will NOT be stored anywhere."
)

input_type = st.radio(
    "Select Input Type",
    ["Image Upload", "Video Upload", "Live Camera"]
)

# ================= IMAGE =================
if input_type == "Image Upload":
    img_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if img_file:
        image = Image.open(img_file).convert("RGB")
        image_np = np.array(image)

        output, face_count = process_image(image_np)
        st.image(output, caption=f"Detected {face_count} face(s)")

# ================= VIDEO =================
elif input_type == "Video Upload":
    video_file = st.file_uploader("Upload a small video", type=["mp4", "avi", "mov"])

    if video_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_path)
        frame_count = 0

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:  # limit frames
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, _ = process_image(frame)
            stframe.image(output)

            frame_count += 1

        cap.release()
        os.remove(temp_path)

# ================= LIVE CAMERA =================
elif input_type == "Live Camera":
    st.warning("Allow camera access when browser asks")

    cam = cv2.VideoCapture(0)
    stframe = st.empty()

    stop = st.button("Stop Camera")

    while cam.isOpened() and not stop:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, _ = process_image(frame)
        stframe.image(output)

    cam.release()

# ================= DATA STORAGE LOGIC =================
if consent:
    st.success("✅ Consent given — data CAN be stored (if backend is connected)")
else:
    st.warning("❌ No consent — data will NOT be stored")



# import streamlit as st
# import cv2
# import torch
# import numpy as np
# import dlib # Make sure dlib is installed
# from PIL import Image
# import torchvision.transforms.functional as TF
# import torch.nn as nn
# import torchvision.models as models

# # 1. Model Definition (Must be identical to the training script)
# class Network(nn.Module):
#     def __init__(self, num_classes=136):
#         super().__init__()
#         self.model_name = 'resnet18'
#         self.model = models.resnet18() # No pretrained=True here, as we load our own weights
#         # Modify the first convolutional layer for single-channel grayscale input
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # Modify the final fully connected layer for 136 landmark outputs
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.model(x)
#         return x

# # 2. Load Model and Dlib Detector
# @st.cache_resource # Cache the model loading for better performance
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Network().to(device)
#     model.load_state_dict(torch.load(r"C:\Users\chall\OneDrive\Desktop\final project\face_landmarks.pth", map_location=device))
#     model.eval()
#     detector = dlib.get_frontal_face_detector()
#     return model, device, detector

# model, device, detector = load_model()

# # 3. Inference Function
# def predict_landmarks(image_pil, model, detector, device):
#     # Convert PIL image to OpenCV format (grayscale)
#     image_np = np.array(image_pil.convert('L'))

#     # Use dlib to detect faces
#     detects = detector(image_np, 1)

#     if len(detects) == 0:
#         st.warning("No face detected. Please try another image.")
#         return None, None

#     # Take the first detected face
#     intersect = detects[0]
#     x1, y1, x2, y2 = intersect.left(), intersect.top(), intersect.right(), intersect.bottom()

#     # Ensure coordinates are within image bounds
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(image_np.shape[1], x2)
#     y2 = min(image_np.shape[0], y2)

#     # Crop the face from the original PIL image
#     cropped_face_pil = image_pil.crop((x1, y1, x2, y2))

#     # Resize to model input size (224x224)
#     input_image_pil = cropped_face_pil.resize((224, 224))

#     # Preprocess the image for the model
#     image_tensor = TF.to_tensor(input_image_pil.convert('L')) # Convert to grayscale tensor
#     image_tensor = TF.normalize(image_tensor, [0.5], [0.5]) # Normalize to [-0.5, 0.5]
#     image_tensor = image_tensor.unsqueeze(0).to(device) # Add batch dimension

#     # Get predictions
#     with torch.no_grad():
#         predictions = model(image_tensor)

#     # Denormalize predictions
#     # Landmarks were normalized to [-0.5, 0.5] relative to 224x224
#     predictions = (predictions.cpu().view(-1, 2) + 0.5) * 224

#     # Adjust landmarks to original cropped face size before final image display
#     # We need to scale landmarks from 224x224 back to the cropped_face_pil dimensions
#     scale_x = cropped_face_pil.width / 224.0
#     scale_y = cropped_face_pil.height / 224.0
#     predictions[:, 0] = predictions[:, 0] * scale_x
#     predictions[:, 1] = predictions[:, 1] * scale_y

#     return cropped_face_pil, predictions.numpy() # Return PIL image and numpy landmarks

# # Streamlit UI
# st.title("Facial Landmark Detection")
# st.write("Upload an image or use your webcam to detect facial landmarks.")

# # Input choice
# input_choice = st.radio("Choose input method:", ("Upload Image", "Webcam"))

# image_file = None
# if input_choice == "Upload Image":
#     image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
# elif input_choice == "Webcam":
#     st.warning("Webcam support in Streamlit Cloud is experimental. For local deployment, ensure you have permissions.")
#     st.markdown("Due to Colab environment limitations, webcam functionality might not work directly within Streamlit hosted here. For local deployment, use `streamlit run app.py` and access from your browser.")
#     # This part would typically be more complex for a robust webcam integration in pure Streamlit
#     # For simplicity, we will disable active webcam capture in Colab
#     st.write("Please upload an image instead for Colab environment.")
#     image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="webcam_fallback")

# if image_file is not None:
#     # Read the image
#     image_pil = Image.open(image_file)
#     st.image(image_pil, caption='Original Image', use_column_width=True)

#     st.subheader("Processing...")

#     # Perform prediction
#     cropped_face_pil, landmarks = predict_landmarks(image_pil, model, detector, device)

#     if cropped_face_pil is not None and landmarks is not None:
#         # Convert cropped PIL image to numpy for drawing
#         cropped_face_np = np.array(cropped_face_pil.convert('RGB'))

#         # Draw landmarks
#         for (x, y) in landmarks:
#             cv2.circle(cropped_face_np, (int(x), int(y)), 2, (0, 255, 0), -1) # Green dots

#         st.image(cropped_face_np, caption='Detected Face with Landmarks', use_column_width=True)


# # import streamlit as st
# # import cv2
# # import torch
# # import numpy as np
# # try:
# #     import dlib
# #     _HAS_DLIB = True
# # except Exception:
# #     dlib = None
# #     _HAS_DLIB = False
# # from PIL import Image
# # import torchvision.transforms.functional as TF
# # import torch.nn as nn
# # import torchvision.models as models

# # # ================= MODEL =================
# # class Network(nn.Module):
# #     def __init__(self, num_classes=136):
# #         super().__init__()
# #         self.model = models.resnet18()
# #         self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
# #         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

# #     def forward(self, x):
# #         return self.model(x)

# # @st.cache_resource
# # def load_model():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = Network().to(device)
# #     model.load_state_dict(torch.load(r"C:\Users\chall\OneDrive\Desktop\final project\face_landmarks1.pth", map_location=device))
# #     model.eval()
# #     if _HAS_DLIB:
# #         detector = dlib.get_frontal_face_detector()
# #     else:
# #         detector = None
# #     return model, detector, device

# # model, detector, device = load_model()

# # # ================= UI =================
# # st.title("Facial Landmark Detection")
# # st.write("Upload a face image")

# # uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# # if uploaded:
# #     image = Image.open(uploaded).convert("RGB")
# #     st.image(image, caption="Original Image")

# #     gray = np.array(image.convert("L"))
# #     if _HAS_DLIB and detector is not None:
# #         faces = detector(gray, 1)
# #         if len(faces) == 0:
# #             st.error("No face detected")
# #             faces = []
# #     else:
# #         cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# #         rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# #         faces = []
# #         class _Rect:
# #             def __init__(self, x, y, w, h):
# #                 self._x = x
# #                 self._y = y
# #                 self._w = w
# #                 self._h = h
# #             def left(self):
# #                 return self._x
# #             def top(self):
# #                 return self._y
# #             def right(self):
# #                 return self._x + self._w
# #             def bottom(self):
# #                 return self._y + self._h
# #         for (x, y, w, h) in rects:
# #             faces.append(_Rect(x, y, w, h))

# #     if len(faces) == 0:
# #         st.error("No face detected")
# #     else:
# #         face = faces[0]
# #         x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
# #         crop = gray[y1:y2, x1:x2]
# #         crop = cv2.resize(crop, (224, 224))

# #         tensor = torch.tensor(crop).unsqueeze(0).unsqueeze(0).float()
# #         tensor = (tensor / 255.0 - 0.5) / 0.5
# #         tensor = tensor.to(device)

# #         with torch.no_grad():
# #             preds = model(tensor)

# #         landmarks = (preds.cpu().view(-1, 2) + 0.5) * 224

# #         fig = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
# #         for (x, y) in landmarks:
# #             cv2.circle(fig, (int(x), int(y)), 1, (255, 0, 0), -1)

# # #         st.image(fig, caption="Predicted Landmarks")
# # # import streamlit as st
# # # import cv2
# # # import numpy as np
# # # from PIL import Image

# # # st.title("Facial Landmark Detection (Verification Mode)")
# # # st.write("This version always produces output to verify the pipeline.")

# # # uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # # if uploaded:
# # #     image = Image.open(uploaded).convert("RGB")
# # #     st.image(image, caption="Original Image", use_column_width=True)

# # #     # Convert to grayscale
# # #     gray = np.array(image.convert("L"))

# # #     # Resize to fixed size
# # #     face = cv2.resize(gray, (224, 224))

# # #     # Generate dummy landmarks (136 values → 68 (x,y))
# # #     landmarks = np.random.rand(68, 2) * 224

# # #     # Draw landmarks
# # #     output = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
# # #     for (x, y) in landmarks:
# # #         cv2.circle(output, (int(x), int(y)), 2, (255, 0, 0), -1)

# # #     st.image(output, caption="Predicted Landmarks (Dummy Output)", use_column_width=True)
