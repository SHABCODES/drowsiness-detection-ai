import urllib.request
import os
import ssl

# Download dlib's facial landmark detector
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("Downloading facial landmark detector...")
    # Bypass SSL verification for this download
    context = ssl._create_unverified_context()
    print("Connecting...")
    with urllib.request.urlopen(url, context=context) as response, \
         open("shape_predictor_68_face_landmarks.dat.bz2", 'wb') as out_file:
        out_file.write(response.read())
    
    # Extract
    import bz2
    print("Extracting...")
    with bz2.open('shape_predictor_68_face_landmarks.dat.bz2', 'rb') as fr, \
         open('shape_predictor_68_face_landmarks.dat', 'wb') as fw:
        fw.write(fr.read())
    print("Download complete!")
