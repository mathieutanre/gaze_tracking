import time
import os
import json
import subprocess
import sys
import pickle
import numpy as np
import torch
import concurrent.futures
import cv2
import glob
import tqdm
import pyaudio
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from images_framework.alignment.opal23_headpose.test.opal23_headpose_test import process_frame, main as main_headpose
from images_framework.src.composite import Composite
from images_framework.src.constants import Modes
from images_framework.alignment.opal23_headpose.src.opal23_headpose import Opal23Headpose
from images_framework.src.viewer import Viewer
from pathlib import Path
import shutil

from facenet_pytorch import MTCNN, InceptionResnetV1


sys.path.append(os.path.join(os.path.dirname(__file__), 'Light-ASD'))

FPS = 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE // FPS 


# Function to run the script in real-time
def run_realtime_script(asd_model_path, cap, stream, p, video_frames, audio_frames, camera_index):
    # Direct call to the main function of realtime.py instead of using subprocess
    import realtime

    # Call the main function of the realtime script
    code_nb_people_detected = realtime.main(asd_model_path, cap, stream, p, video_frames, audio_frames, camera_index)
    print("realtime.py executed successfully.")
    return code_nb_people_detected


def analyze_scores(camera_index):
    scores_file = os.path.join(os.path.dirname(__file__), 'demo', str(camera_index), 'pywork', 'scores.pckl')
    
    if not os.path.exists(scores_file):
        print("Scores file not found.")
        return None

    with open(scores_file, 'rb') as fichier:
        scores = pickle.load(fichier)

    if not scores:
        print("No scores found in the file.")
        return None

    # None if person_score is empty
    moyennes = [np.mean(person_score) if len(person_score) > 0 else None for person_score in scores]

    if all(score is None for score in moyennes):
        print("All scores are None.")
        return None

    # Filter out None values for the purpose of finding the max
    valid_moyennes = [(index, score) for index, score in enumerate(moyennes) if score is not None]

    if len(valid_moyennes) > 0:
        print(valid_moyennes)

        max_index, max_average_score = max(valid_moyennes, key=lambda x: x[1])
        
        # If the maximum of the valid averages is negative, set max_index to -1
        if max_average_score < 0:
            max_index = -1
    else:
        max_index = -1
        max_average_score = None

    return max_index, max_average_score


def capture_camera(cap, asd_model_path, stream, p, video_frames, audio_frames, camera_index, json_folder):
    if not cap.isOpened():
        print(f"Error: Unable to open the webcam {camera_index}")
        return

    code_nb_people_detected = run_realtime_script(asd_model_path, cap, stream, p, video_frames, audio_frames, camera_index)

    if code_nb_people_detected == 300: # More than 2 people were detected

        max_index, max_score = analyze_scores(camera_index)

    elif code_nb_people_detected == 100: # 0 people was detected

        max_index, max_score = 100, None
    
    elif code_nb_people_detected == 200: # 1 people was detected

        max_index, max_score = 200, None

    return max_index, max_score


def calculate_angle_no_as(camera_indices):

    # camera_indices should be a list of lists with the camera indices and streams [[cap_0, 0], [cap_1, 1], [cap_2, 2], ...]
    
    def process_camera(camera_index):
        try:
            # Executes the main_headpose function for each camera
            main_headpose(cap=camera_index[0], show_viewer=False, save_image=True, database='300wlp', gpu=-1, rotation_mode='euler', camera_index=camera_index[1])
        except Exception as e:
            print(f"An error occurred while processing camera {camera_index[1]}: {e}")
    
    # Using ThreadPoolExecutor to parallelize the processing of both cameras
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_camera, camera_index) for camera_index in camera_indices]
        
        # Iterate through the results as tasks are completed
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred while processing a camera: {e}")


def process_yaw_angles_all_json(json_folders):
    # Create a list of lists to store yaw angles
    yaw_list = [[] for _ in range(len(json_folders))]

    # Loop through each folder in json_folders
    for i in range(len(json_folders)):
        # Loop through all JSON files in the folder
        for filename in os.listdir(json_folders[i]):
            if filename.endswith('.json'):
                json_path = os.path.join(json_folders[i], filename)
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Iterate through all annotations to extract yaw angles
                    for annotation in data.get('annotations', []):
                        yaw = annotation['pose'][0]  # Extract the first element of 'pose' (yaw angle)
                        yaw_list[i].append(yaw)

    # Calculate the average yaw angle for each sublist
    yaw_averages = [sum(sublist) / len(sublist) if sublist else 180 for sublist in yaw_list]

    return yaw_averages


## Multiple people detected and one active speaker

# Initialization of MTCNN for face detection and FaceNet for embeddings
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def apply_facenet_to_cropped_face(cropped_face):
    """
    Applies FaceNet to extract embeddings from a frame already cropped to a face.
    Returns a tuple (embeddings, bbox), where bbox is a dummy box (0, 0, width, height).
    """
    # Check if the cropped image is not empty before proceeding
    if cropped_face is None or cropped_face.size == 0:
        return None
    
    # Resize the face for FaceNet if necessary
    cropped_face = cv2.resize(cropped_face, (160, 160))  # Resize for FaceNet

    # Convert the face to a tensor for FaceNet
    face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Extract embeddings with FaceNet
    with torch.no_grad():
        embedding = facenet(face_tensor)

    # The dummy bounding box is (0, 0, width, height) because the face is already cropped
    bbox = (0, 0, cropped_face.shape[1], cropped_face.shape[0])

    # Return the embeddings and the dummy bounding box
    return (embedding, bbox)


def apply_facenet(frame):
    """
    Applies MTCNN to detect faces in a frame and extracts embeddings with FaceNet.
    Returns a list of tuples (embeddings, bbox) for each detected face.
    """
    # Convert the frame to PIL format because MTCNN expects PIL images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection with MTCNN
    boxes, _ = mtcnn.detect(frame_rgb)
    
    # If no face is detected, return an empty list
    if boxes is None or len(boxes) == 0:
        return []
    
    embeddings = []
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Ensure the coordinates are valid before cropping
        if x1 < 0 or y1 < 0 or x2 > frame_rgb.shape[1] or y2 > frame_rgb.shape[0]:
            continue
        
        cropped_face = frame_rgb[y1:y2, x1:x2]  # Extract the face
        
        # Check if the cropped image is not empty before resizing
        if cropped_face.size == 0:
            continue
        
        cropped_face = cv2.resize(cropped_face, (160, 160))  # Resize for FaceNet

        # Convert the face to a tensor for FaceNet
        face_tensor = torch.tensor(cropped_face).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Extract embeddings with FaceNet
        with torch.no_grad():
            embedding = facenet(face_tensor)
        
        embeddings.append((embedding, (x1, y1, x2 - x1, y2 - y1)))  # Add the face coordinates (bbox)

    return embeddings


def find_matching_face(target_face, detected_faces):
    """
    Compares the target face with the detected faces and returns the face that best matches.
    Uses the Euclidean distance between embeddings to measure similarity.
    """
    target_embedding, _ = target_face
    
    best_match = None
    min_distance = float('inf')
    
    for embedding, bbox in detected_faces:
        # Calculate the Euclidean distance between the target face embeddings and the detected face embeddings
        distance = torch.dist(target_embedding, embedding).item()
        
        # Find the face with the smallest distance (best match)
        if distance < min_distance:
            min_distance = distance
            best_match = bbox  # Save the coordinates of the matching face

    # If a match is found, return the face coordinates
    return best_match


def crop_to_face(frame, face_coordinates):
    """
    Crops the image around the face coordinates.
    Returns the cropped image.
    """
    x, y, w, h = face_coordinates
    return frame[y:y+h, x:x+w]


def calculate_angle(max_index_person, caps, composite, viewer, dirnames):
    # Path to the AVI file corresponding to max_index_person
    avi_file_path = os.path.join('demo', '0', 'pycrop', f'{max_index_person:05d}.avi')

    # Load the first image from the corresponding AVI file
    capture = cv2.VideoCapture(avi_file_path)
    success, target_frame = capture.read()
    capture.release()

    if not success:
        raise ValueError(f"Unable to read from AVI file: {avi_file_path}")

    # Apply FaceNet to the target image to identify the person's face
    target_face = apply_facenet_to_cropped_face(target_frame)

    def process_camera_frames(cap, dirname):
        cropped_frames = []
        last_frame = None
        matched_face_coords = None

        # Capture 3 frames from the camera
        for _ in range(3):
            success, frame = cap.read()
            if success:
                # Identify faces in each frame with FaceNet
                detected_faces = apply_facenet(frame)
                
                # Compare the detected faces with the target face
                matched_face = find_matching_face(target_face, detected_faces)
                
                if matched_face is not None:
                    # Crop the frame to keep only the person's head
                    cropped_frame = crop_to_face(frame, matched_face)
                    cropped_frames.append(cropped_frame)
                    last_frame = frame  # Save the last frame
                    matched_face_coords = matched_face  # Save the coordinates of the matched face

        # Call process_frame for each cropped frame
        for idx, cropped_frame in enumerate(cropped_frames):
            # Temporarily save the cropped image for processing
            temp_filename = os.path.join(dirname, f'cropped_frame_{idx}.png')
            cv2.imwrite(temp_filename, cropped_frame)
            
            # Call process_frame to calculate the face angle
            process_frame(composite, temp_filename, show_viewer=False, save_image=True, viewer=viewer, delay=1, dirname=dirname)

        return last_frame, matched_face_coords

    # Use ThreadPoolExecutor to parallelize the processing of cameras
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_camera_frames, caps[i], dirnames[i]) for i in range(len(caps))]

        # Wait for all cameras to finish processing and obtain the results
        results = [f.result() for f in futures]


    # Adapt the audio
    last_frame, matched_face_coords = results[0]
    if last_frame is not None and matched_face_coords is not None:
        x, y, w, h = matched_face_coords
        # Calculate the horizontal position of the face as a percentage of the frame width
        frame_width = last_frame.shape[1]
        face_center_x = x + w / 2
        horizontal_position = face_center_x / frame_width  # Value between 0 (left) and 1 (right)

        # Calculate audio volume distribution based on horizontal position
        left_volume = (1 - horizontal_position) * 100
        right_volume = horizontal_position * 100

        print(f"Audio Output: Left Speaker = {left_volume:.2f}%, Right Speaker = {right_volume:.2f}%")
    
    
    # Only for testing, remove later

    output_dir = './highlighted_speaker'
    os.makedirs(output_dir, exist_ok=True)

    for i, (last_frame, matched_face_coords) in enumerate(results):
        if last_frame is not None and matched_face_coords is not None:
            x, y, w, h = matched_face_coords
            # Create a spotlight effect to highlight the detected face
            mask = np.zeros_like(last_frame, dtype=np.uint8)
            cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (255, 255, 255), -1)
            highlighted_frame = cv2.addWeighted(last_frame, 0.7, mask, 0.3, 0)

            # Save the modified image
            output_file = os.path.join(output_dir, f'highlighted_speaker_camera_{i}.png')
            cv2.imwrite(output_file, highlighted_frame)

    print(f"Angle calculation completed for person index {max_index_person}.")


def main(num_cameras=3):
    # Initialize audio capture
    p = pyaudio.PyAudio()

    asd_model_path = 'asd_model.pth'
    json_folders = [f'./images_framework/output/{i}/images' for i in range(num_cameras)]

    # Initialize the webcams dynamically based on num_cameras
    caps = []
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if not cap.isOpened():
            print(f"Error: Unable to open webcam {i}")
            return
        caps.append(cap)

    current_displayed_camera = 0  # Start with the first camera displayed

    # Display the first camera before entering the loop
    ret, frame = caps[0].read()
    if ret:
        cv2.imshow("Camera Display", frame)
        cv2.waitKey(1000)  # Wait for 1 second to display the initial frame

    execution_times = []
    average_angles = []
    average_scores = []

    for iteration in range(10):
        # Initialize video and audio frames for this iteration
        video_frames = [deque() for _ in range(num_cameras)]
        audio_frames = deque()

        # Initialize the audio stream for each iteration
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        start_time = time.time()

        # Capture for all cameras
        max_index_person, max_score = capture_camera(caps[0], asd_model_path, stream, p, video_frames[0], audio_frames, 0, json_folders[0])
        print(f'Max score : {max_score}')
        average_scores.append(max_score)

        if max_index_person == 100:  # Nobody was detected
            continue  # Go to next iteration

        elif max_index_person == 200 or max_index_person == -1:  # 1 person or more detected, but no active speaker
            calculate_angle_no_as([(caps[i], i) for i in range(num_cameras)])  # Process Headpose detection for all cameras
            yaw_averages = process_yaw_angles_all_json(json_folders)
            print(f'YAW AVERAGE : {yaw_averages}')

        else:
            # Configure composite with additional options
            composite = Composite()
            sr = Opal23Headpose('images_framework/alignment/opal23_headpose/')
            composite.add(sr)
            composite.parse_options(['--database', '300wlp', '--gpu', '-1', '--rotation-mode', 'euler'])
            composite.load(Modes.TEST)

            # Directory to save frames for all cameras
            dirnames = [f'images_framework/output/{i}/images/' for i in range(num_cameras)]
            for dirname in dirnames:
                Path(dirname).mkdir(parents=True, exist_ok=True)

            # Call calculate_angle for headpose detection for the person with max_index_person
            calculate_angle(max_index_person, caps, composite, Viewer('opal23_headpose_test'), dirnames)
            yaw_averages = process_yaw_angles_all_json(json_folders)

        # Find the camera with the average angle closest to 0 (ignoring None values)
        closest_camera_index = None
        closest_angle = float('inf')

        for i in range(num_cameras):
            if yaw_averages[i] is not None and abs(yaw_averages[i]) < closest_angle:
                closest_camera_index = i
                closest_angle = abs(yaw_averages[i])

        # If a valid closest camera is found, update the displayed camera
        if closest_camera_index is not None and closest_camera_index != current_displayed_camera:
            current_displayed_camera = closest_camera_index
            print(f"Displaying camera {current_displayed_camera} with angle closest to 0: {closest_angle}")

        # Display logic for selected camera
        if max_index_person == 200 or max_index_person == -1:
            def get_first_tif_image(image_dir):
                # List all .tif files in the directory
                for file in os.listdir(image_dir):
                    if file.endswith('.tif'):
                        image_file = os.path.join(image_dir, file)
                        return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
                return None

            image_paths = [f'./images_framework/output/{i}/images' for i in range(num_cameras)]
            frame = get_first_tif_image(image_paths[current_displayed_camera])

            if frame is not None:
                cv2.imshow("Camera Display", frame)
                cv2.waitKey(1)
            else:
                print(f"Unable to load image from directory.")

        else:  # Active speaker
            folder_path = './highlighted_speaker'
            image_file = os.path.join(folder_path, f'highlighted_speaker_camera_{current_displayed_camera}.png')
            frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

            if frame is not None:
                cv2.imshow("Camera Display", frame)
                print('display image')
                cv2.waitKey(1)
            else:
                print(f"Unable to load highlighted image from {folder_path}")

        # Close the stream and other resources at the end of each iteration
        stream.stop_stream()
        stream.close()

        end_time = time.time()
        execution_times.append(end_time - start_time)
        print(f"Iteration {iteration + 1} completed in {execution_times[-1]:.2f} seconds")
        print(yaw_averages)
        average_angles.append(yaw_averages)

        # Delete json_folders
        for folder in json_folders:
            shutil.rmtree(folder)
            os.makedirs(folder)

        # Delete Demo
        shutil.rmtree('./Demo')
        os.makedirs('./Demo')

    print(f"All iterations done")
    print(f"Execution times: {execution_times}")
    print(f"Average angles for all iterations: {average_angles}")
    print(f"Average scores for all iterations: {average_scores}")

    # Clean up everything at the end
    for cap in caps:
        cap.release()
    p.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(num_cameras=2)  # Change this to 2 or 3 depending on the number of cameras you have

