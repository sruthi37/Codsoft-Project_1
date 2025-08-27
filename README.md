**Name     :** Sruthi R

**Domain   :** Artificial Intelligence


## OVERVIEW OF THE PROJECT


### PROJECT : AI FACE DETECTION AND RECOGNITION APPLICATION USING PRE-TRAINED FACE DETECTIONS MODELS LIKE HAAR CASCADES, SIAMESE NETWORKS OR ARCFACE





### OBJECTIVE



The primary objective of this project is to develop an AI-based application that can **detect and recognize human faces** in images or real-time video streams. This involves using **computer vision and deep learning techniques** to accurately locate and identify faces.  

The project aims to:  

>**DETECT FACES IN IMAGES OR VIDEOS**  
   - Utilize **pre-trained face detection models** like **Haar cascades** or **deep learning-based detectors** (such as OpenCV’s DNN module or MTCNN).  
   - Process video streams in real time to identify and highlight faces.  

   
>**ENHANCE FACE DETECTION ACCURACY**  
   - Improve detection performance using advanced techniques like **MobileNet SSD** or **YOLO** for better speed and accuracy.  
   - Optimize detection in various lighting conditions and angles.  


>**IMPLEMENT FACE DETECTION (Optional)**  
   - Use deep learning techniques such as **Siamese Networks** or **ArcFace** to recognize and differentiate individuals.  
   - Train the model to identify and verify faces from a given dataset.  


>**PROVIDE REAL-TIME PROCESSING**  
   - Integrate the solution with a webcam or external camera to detect and recognize faces in real time.  
   - Apply performance optimizations to ensure smooth and efficient execution.  


>**DEVELOP A FUNCTIONAL & SCALABLE SYSTEM**  
   - Ensure that the face detection and recognition system can be used for various applications like **security, authentication, attendance systems, and AI-based surveillance.**  
   - Implement a user-friendly interface for testing and deployment.  
  
This project will result in a **working AI-powered face detection and recognition application** that can accurately identify faces in images or video streams, with potential real-world applications in security, access control, and smart surveillance.





### KEY ACTIVITIES


 
>**SETUP ENVIRONMENT**  
   - Install **OpenCV, NumPy, dlib, face-recognition**.  
   - Install **CMake** for `dlib` compatibility.  


>**LOAD IMAGES & VIDEO**  
   - Read images/videos using OpenCV.  
   - Preprocess (grayscale conversion, resizing).  


>**FACE DETECTION**  
   - Use **Haar cascades** or deep learning detectors.  
   - Detect faces in images/videos in real time.  


>**FACE RECOGNITION (Optional)**  
   - Use **Siamese Networks, ArcFace**, or `face-recognition`.  
   - Identify and label detected faces.  


>**REAL-TIME PROCESSING**  
   - Capture and process webcam feed.  
   - Apply detection and recognition in live video.  


>**APPLY AI FILTERS (Optional)**  
   - Overlay sunglasses or cartoon effects.  
   - Enhance visualization with creative filters.  


>**DEBUGGING & OPTIMIZATION**  
   - Fix errors and improve accuracy.  
   - Optimize performance for real-time use.  


>**DEPLOYMENT & FUTURE ENHANCEMENTS**  
   - Expand for **security, mobile, or web apps**.  
   - Add more advanced AI-based recognition models.





### TOOLS AND TECHNOLOGIES USED 



>**PROGRAMMING LANGUAGE**  
   - **Python**: Used for writing the entire AI application due to its strong libraries and frameworks for machine learning and image processing.  


>**LIBRARIES AND FRAMEWORKS**  
   - **OpenCV(C++ Backend)** : For Image processing, face detection, and applying filters. Internally uses optimized C++ functions.
   - **NumPy**  : For numerical computations, handling image matrices.  
   - **dlib**   : Provides deep learning-based face detection models and face landmarks.  
   - **face-recognition**: A simple and powerful library built on dlib for face recognition.  


>**MACHINE LEARNING & AI TECHNIQUES**  
   - **Haar Cascades**: A pre-trained model for face detection based on feature matching.  
   - **Deep Learning-Based Face Detectors**: More accurate than Haar cascades, using CNNs.  
   - **Siamese Networks / ArcFace (Optional)**: For high-accuracy face recognition.
     

>**DEVELOPMENT ENVIRONMENT**  
   - **Visual Studio Code (VS Code)**: Used for writing and debugging Python scripts.  
   - **Jupyter Notebook (Optional)** : For testing face detection/recognition models.  


>**SYSTEM REQUIREMENTS & DEPENDENCIES**  
   - **Python 3.x**: Must be installed for running the scripts.  
   - **CMake**: Needed for compiling dlib (especially for Windows users).  
   - **Pip Package Manager**: To install required libraries.  


>**HARDWARE REQUIREMENTS**  
   - **Webcam**: For real-time face detection and recognition.  
   - **GPU (Optional)**: If deep learning-based models are used for better performance.  


>**COMMAND LINE TOOLS**  
   - **Command Prompt (CMD) / Terminal**: To run Python scripts, install libraries, and debug errors.  


>**IMAGE & VIDEO SOURCES**  
   - **Pre-stored Images**: Used for testing detection and recognition.  
   - **Live Video Feed**: Captured from a webcam for real-time application.  


>**OPTIONAL ENHANCEMENTS**  
   - **Augmented Reality (AR) Filters**: Applying sunglasses, cartoon effects, etc., using OpenCV overlays.  
   - **Edge Detection & Image Processing**: For better visualization of facial features.  

This project combines **Computer vision, deep learning, and AI** techniques to create a real-time face detection and recognition system. 





### IMPLEMENTATION



>**INSTALL DEPENDENCIES**

Use CMD to install required libraries:

    pip install opencv-python numpy dlib face-recognition cmake
    

    
>**FACE DETECTION USING OPENCV HAAR CASCADES**
   
    import cv2

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

>RUN IN CMD:

    python face_detection.py



>**FACE RECOGNITION (USING face-recognition LIBRARY)**
    
    import face_recognition
    import cv2

    known_image = face_recognition.load_image_file("known_face.jpg")
    known_encoding = face_recognition.face_encodings(known_image)[0]
    known_faces = [known_encoding]
    known_names = ["Person 1"]

    cap = cv2.VideoCapture(0)  

    while True:
             ret, frame = cap.read()
             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             face_locations = face_recognition.face_locations(rgb_frame)
             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

             for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                 matches = face_recognition.compare_faces(known_faces, face_encoding)
                 name = "Unknown"
                 if True in matches:
                     name = known_names[matches.index(True)]

                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                 cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


             cv2.imshow("Face Recognition", frame)
             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
    cap.release()
    cv2.destroyAllWindows()         

>RUN IN CMD:

    python face_recognition.py



>**REAL-TIME FACE DETECTION(USING PRE-TRAINED FACE DETECTION MODEL)**

    import cv2

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
    'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)  # 0 for default webcam

    while True:
        ret, frame = cap.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

>RUN IN CMD:

    python real_time_face_detection.py   



>**RUNNING THE LIVE AI FILTERS(optional)**

    sunglasses = cv2.imread("sunglasses.png", -1)
    
    def add_sunglasses(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sunglasses_resized = cv2.resize(sunglasses, (w, int(h / 3)))
            sw, sh, _ = sunglasses_resized.shape

            for i in range(sw):
                for j in range(sh):
                    if sunglasses_resized[i, j, 3] != 0:
                        frame[y + i, x + j] = sunglasses_resized[i, j, :-1]

            return frame

>RUN IN CMD:

    python live_filter.py

This project **Detects**, **Recognizes**, and **Applies filters** to faces in real-time using **OpenCV and face-recognition**.





### HOW IT WORKS?


This project detects faces in real-time and applies AI filters using computer vision techniques. It uses OpenCV and deep learning-based models to process images or videos and identify human faces accurately.



### FACE DETECTION (USING OPENCV HAAR CASCADES & DEEP LEARNING MODELS)

**How it works?** 


>**CAPTURE LIVE VIDEO OR LOAD AN IMAGE**  
   - The webcam is activated using OpenCV (`cv2.VideoCapture()`), or an image is loaded from storage.  
   - Each video frame or image is processed separately in a loop.  


>**CONVERT THE IMAGE TO GRAYSCALE**  
   - Since Haar cascades work better on grayscale images, we convert the input image using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.  


>**APPLY FACE DETECTION ALGORITHM**  
   - The face detection model scans the image and identifies **regions that likely contain a face** based on patterns.  
   - A bounding box is drawn around the detected face using `cv2.rectangle()`.  


>**DISPLAY THE PROCESSED FRAME**  
   - The detected face is highlighted, and the processed frame is displayed in a window.  
   - The system keeps detecting faces in live video until the user presses the **'q'** key to quit.  


>**Key Techniques Used:**  
   - **Haar Cascade Classifier** (Pre-trained XML model)  
   - **Deep Learning-Based Face Detector** (Optional for better accuracy)  


>**CMD Command to Run:**  

    python face_detection.py
 


### FACE RECOGNITION (If Implemented in Future)    

**How Face Recognition Works?** 

In this project, I did not implement face recognition. However, if added, face recognition would work as follows:  


>**LOAD KNOWN FACES FROM A DATASET**  
   - Face images of known individuals are loaded and encoded into numerical representations.  


>**EXTRACT FACE ENCODINGS FROM LIVE VIDEO**  
   - The system detects faces in real-time video and converts them into numerical encodings using a deep learning model.  


>**COMPARE ENCODINGS WITH STORED FACES**  
   - The newly detected face is compared with stored encodings to check for a match.  
   - If the match confidence is above a threshold, the face is **recognized** and the person’s name is displayed.  


>**DISPLAY RESULTS ON SCREEN**  
   - The recognized face is marked with a **bounding box** and the **name of the person** appears.  


>**KEY TECHNIQUES USED (If Implemented):**  
- Deep Learning Face Embeddings (e.g., **ArcFace, FaceNet, or dlib**)  
- Face Distance Calculation for Recognition  


>**CMD Command to Run(If Implemented):**  

    python face_recognition.py



### LIVE AI FILTERS (Fun Feature for Augmented Reality) 

**How It Works?**  


>**DETECT FACE IN LIVE VIDEO**  
   - The face is detected in real-time using OpenCV’s Haar cascade classifier.
     

>**LOAD AN IMAGE FILTER (Example: Sunglasses, Hat, Mask, etc.)**  
   - The overlay filter (e.g., **sunglasses.png**) is loaded using `cv2.imread()`.  
   - The filter must support transparency (RGBA format).  


>**RESIZE AND OVERLAY THE FILTER ONTO THE FACE**  
   - The filter image is resized to match the face dimensions.  
   - A loop runs through every pixel of the filter, overlaying it onto the detected face while maintaining transparency.  


>**DISPLAY THE LIVE VIDEO WITH AI FILTER APPLIED**  
   - The processed video stream is displayed, showing the person wearing the AI filter.  


>**KEY TECHNOLOGIES USED:**  
   - **Image Overlaying (Alpha Blending)** 
   - **Face Landmark Detection (for better placement of filters)**  


>**CMD COMMAND TO RUN:**  

    python live_filter.py

This project successfully detects and enhances human faces in real-time, using AI and image processing techniques.





### OUTPUT


1)

![Screenshot 2025-03-11 133637](https://github.com/user-attachments/assets/946deb3c-4bfa-4ae1-b1a8-d755a2fe464d)

The code of **test_opencv.py & face_detection.py** is executed in **VS build tools** and the above screenshot is about the command I used in the **CMD Prompt**, to get the expected output.


![Screenshot 2025-03-11 133728](https://github.com/user-attachments/assets/d0494dae-7323-4ee0-8eb8-1c544703374d)

The above screenshot is the expected output of the face_detection.py. Here the image's face is being identified by covering the face with the **Green Rectangular box**, after executing the code in **VS Biuld Tools** and the command 'python face_detection.py' in **CMD**.



2)

![Screenshot 2025-03-11 133920](https://github.com/user-attachments/assets/24ba155f-fe37-4135-8493-03ca96dd324d)

The code of **real_time_face_detection.py** is executed in **VS build tools** and the above screenshot is about the command I used in the **CMD Prompt**, to get the expected output.


![Screenshot 2025-03-11 134441](https://github.com/user-attachments/assets/fb00b6c7-03df-4f19-aef2-9d084ce1f5a7)

The above screenshot is the expected output of the **real_time_face_detection.py**. Here the my face is being identified by covering the face with the **Green Rectangular box**, after executing the code in **VS Biuld Tools** and the command 'python real_time_face_detection.py' in **CMD**.



3)

![Screenshot 2025-03-11 134701](https://github.com/user-attachments/assets/b3bfa7c0-e67c-4329-a7fb-8d8b6bc8115d)

The code of **grayscale_filter.py** is executed in **VS build tools** and the above screenshot is about the command I used in the **CMD Prompt**, to get the expected output.


![Screenshot 2025-03-11 135429](https://github.com/user-attachments/assets/b982f7f6-2353-4c85-b5cc-cee7929da5b6)

The above screenshot is the expected output of the **graysacle_filter.py**. Here the total screen is **turned into black and white**, which is the **Grayscale Filter**, after executing the code in **VS Biuld Tools** and the command 'python grayscale_filter.py' in **CMD**.



4)

![Screenshot 2025-03-11 135630](https://github.com/user-attachments/assets/29b799b6-0ea7-490a-8c37-9c55406f2365)

The code of **cartoon_filter.py** is executed in **VS build tools** and the above screenshot is about the command I used in the **CMD Prompt**, to get the expected output.


![Screenshot 2025-03-11 135721](https://github.com/user-attachments/assets/6f2b7be6-5f2b-45a4-a6fe-d2a52b01e61d)

The above screenshot is the expected output of the **cartoon_filter.py**. Here the total screen is **turned into an animated filter**, which is the **Cartoon Filter**, after executing the code in **VS Biuld Tools** and the command 'python cartoon_filter.py' in **CMD**.



5)

![Screenshot 2025-03-11 135834](https://github.com/user-attachments/assets/b3310f72-bcb3-4bcb-a767-b744bbcddac6)

The code of **virtual_sunglasses.py** is executed in **VS build tools** and the above screenshot is about the command I used in the **CMD Prompt**, to get the expected output.


![Screenshot 2025-03-11 135931](https://github.com/user-attachments/assets/a4eac6b7-cf2b-485c-91b7-21f5d6f80378)

The above screenshot is the expected output of the **virtual_sunglasses.py**. Here I have a sunglass, which **overlay in my forehead** and this is the **Virtual Sunglasses Filter**, after executing the code in **VS Biuld Tools** and the command 'python cartoon_filter.py' in **CMD**.





### CONCLUSION 

This project successfully demonstrates the implementation of real-time face detection and AI-based live filters using computer vision and deep learning techniques. By utilizing OpenCV's Haar cascades and deep learning-based face detectors, the system accurately detects human faces in images and videos. Additionally, AI filters, such as sunglasses overlays, enhance the functionality by adding augmented reality (AR) elements to detected faces.

The system works in real-time, making it highly applicable for various use cases, such as security surveillance, user authentication, human-computer interaction, and entertainment-based AR applications. The modularity of the project allows easy integration of advanced deep learning models for better accuracy and robustness in face detection.



>**KEY TAKEWAYS**

- **Accurate Face Detection:** The project efficiently detects faces using both Haar cascades and deep learning-based methods.  
- **Live Video Processing:** The system processes real-time video streams, making it practical for security and interactive applications.  
- **Augmented Reality Features:** The project integrates AI filters that overlay virtual objects, showcasing real-time AR capabilities.  
- **Scalability and Future Enhancements:** The system can be extended to include **face recognition** using **deep learning models like ArcFace or Siamese Networks**, improving security applications.



>**FUTURE IMPROVEMENTS**
 
- **Implementing Face Recognition:** Enhancing the system to recognize and identify individuals in real-time.  
- **Using Advanced Deep Learning Models:** Replacing Haar cascades with CNN-based face detectors like **MTCNN or RetinaFace** for improved accuracy.  
- **Integrating AI Filters for Multiple Faces:** Extending the filtering capability to apply AR effects to multiple detected faces simultaneously.  
- **Optimizing Performance:** Using **GPU acceleration** to enhance real-time video processing speed.

In conclusion, this project provides a **strong foundation for AI-powered face detection and AR-based live filters**, demonstrating the capabilities of **Computer vision, Image processing, and AI in Real-world Applications**. 


  







    
