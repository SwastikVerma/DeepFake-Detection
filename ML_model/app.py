
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import tensorflow as tf
import numpy as np

app = FastAPI()

# Mount a directory to serve static files (e.g., videos)
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_path = 'pretrained_model.keras'
# model = load_model(model_path, compile=False, input_shape=(224, 224, 3))\

from keras.models import load_model

# Define the input shape when creating your model
input_shape = (224, 224, 3)  # Example input shape
model = load_model('10000_model_1.h5')

@app.post("/predict")
async def upload_video(request: Request, file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}

    # Save the uploaded video file
    filename = secure_filename(file.filename)
    with open(os.path.join("videos", filename), "wb") as f:
        f.write(await file.read())
    prediction_frames = video_to_image(filename=filename)
    # if len(prediction_frames) == 0:
    #     temp = {"prediction":"No Face detected in a frame!!!","no_of_frames":0,"accuracy":"NA"}
    #     return JSONResponse(content=temp)
    cntt=0
    cntf=0
    result = 0
    sum = 0
    for img in prediction_frames:
        img_array = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # img_array = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        sum+=predictions
        print("prediction : ",predictions)
        if(predictions <= 0.6) :
            result = result - 1
            cntf+=1
        else :
            result = result + 1
            cntt+=1

        # Print or use the predictions as needed
        # print(predictions)
    
    if cntt>= cntf+5:
        accuracy = (sum/(cntt+cntf))*100
        print("accuracyT:",accuracy)
        response = {"prediction": "Real Video","no_of_frames":len(prediction_frames),"accuracy":accuracy}
    else:
        
        accuracy = (sum/(cntt+cntf))*100
        print("accuracyF:",100-accuracy)
        response = {"prediction": "Fake Video","no_of_frames":len(prediction_frames),"accuracy":accuracy}
        
    # Convert NumPy arrays to Python lists if needed
    for key, value in response.items():
        if isinstance(value, np.ndarray):
            response[key] = value.tolist()

    return JSONResponse(content=response)




def video_to_image(filename):
    video_path = os.path.join("videos", filename)
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return
    prediction_frames=[]
    frame_cnt = 0
    idx =0
    gaps =6
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break
        
        if idx == 0:
            
            temp = face_detection(frame)
            if(type(temp) != type(-1)):
                prediction_frames.append(temp)
                frame_cnt+=1
        else :
            if idx%gaps == 0:
                temp = face_detection(frame)
                if(type(temp) != type(-1)):
                    prediction_frames.append(temp)
                    # print(type(temp))
                    frame_cnt+=1
        idx+=1
    return prediction_frames


import glob,cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detection(f):
    img = f
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image

    if (len(faces) == 0) :
        return -1
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    roi_color = img[y:y+h, x:x+w]  # Extract the region of interest (face)
    # roi_color = cv2.resize(roi_color, (244, 244))
    # roi_color = roi_color/255
    return roi_color



	
	

import nest_asyncio
import uvicorn
from pyngrok import ngrok


if __name__ == "__main__":
    

    ngrok_tunnel = ngrok.connect(8000)
    print("Public Url: " + ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)



