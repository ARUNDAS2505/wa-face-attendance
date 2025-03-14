import time
from flask import Flask, request, jsonify, send_file 
import os
import shutil
import cv2
from deepface import DeepFace
from mtcnn import MTCNN


app = Flask(__name__)

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    try:
        # Step 1: Get the image and db_path from the request
        if 'image' not in request.files or 'db_path' not in request.form:
            return jsonify({"error": "Image file and db_path are required"}), 400

        image_file = request.files['image']
        db_path = request.form['db_path']

        # Step 2: Save the uploaded image to a temporary file
        image_path = "temp_group_photo.jpg"
        image_file.save(image_path)

        # Step 3: Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            return jsonify({"error": "Unable to read the uploaded image."}), 400

        # Step 4: Detect faces using MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image)

        if not faces:
            return jsonify({"message": "No faces detected in the image."}), 200

        # Step 5: Create folder for temp faces and processed image
        temp_faces_folder = "temp_faces"
        processed_image_folder = "processed_images"
        os.makedirs(temp_faces_folder, exist_ok=True)
        os.makedirs(processed_image_folder, exist_ok=True)

        # Set to store unique folder names (person names)
        unique_folders = set()

        # Step 6: Loop through each detected face
        for i, face in enumerate(faces):
            x, y, w, h = face['box']  # MTCNN provides (x, y, width, height)
            face_img = image[y:y+h, x:x+w]  # Crop the face from the image
            face_path = f"{temp_faces_folder}/face_{i+1}.jpg"
            cv2.imwrite(face_path, face_img)  # Save the face

            try:
                # Use enforce_detection=False to prevent the script from crashing
                print(f"Searching for matches for face {i+1}...")
                recognition_list = DeepFace.find(img_path=face_path, db_path=db_path, enforce_detection=False)
                
                if isinstance(recognition_list, list) and recognition_list:
                    for recognition_df in recognition_list:
                        if not recognition_df.empty:
                            # Extract the folder name (person's name) from the recognition path
                            matched_folders = recognition_df['identity'].apply(lambda x: os.path.basename(os.path.dirname(x)))  # Extract folder name only
                            name = matched_folders.mode()[0] if not matched_folders.empty else None
                            if name:
                                unique_folders.add(name)  # Add folder names to the set
                                # Draw a rectangle around the face on the original image and label it
                                center = (x + w // 2, y + h // 2)
                                radius = max(w, h) // 2
                                cv2.circle(image, center, radius, (235, 164, 0), 3)
                                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 164, 0), 3)  # Add name label
                            break

            except ValueError as e:
                print(f"Warning: No face detected in {face_path}. Skipping this image.")

        # Step 7: Save the processed image with squared faces and names
        processed_image_path = os.path.join(processed_image_folder, "processed_group_photo.jpg")
        cv2.imwrite(processed_image_path, image)

        # Step 8: Print unique folder names (person names) at the end
        result_text = "\nAll unique person names found in the group photo:\n" + "\n".join(unique_folders)

        # Step 9: Clean up - Delete temp faces folder and uploaded image
        try:
            shutil.rmtree(temp_faces_folder)
            os.remove(image_path)
            print("\ntemp_faces folder and temp image file deleted successfully.")
        except Exception as e:
            print(f"Error deleting temp files: {e}")

        return jsonify({"result": result_text, "processed_image_path": processed_image_path}), 200
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the image.", "details": str(e)}), 500


@app.route('/register_class', methods=['POST'])
def register_class():
    try:
        # Get the class name from the request
        if 'class_name' not in request.form:
            return jsonify({"error": "Class name is required"}), 400

        class_name = request.form['class_name']

        # Create the folder path in the database directory
        class_folder_path = os.path.join('database', class_name)

        # Check if the folder already exists
        if os.path.exists(class_folder_path):
            return jsonify({"message": f"Class '{class_name}' already exists."}), 200

        # Create the folder
        os.makedirs(class_folder_path, exist_ok=True)

        return jsonify({"message": f"Class '{class_name}' successfully registered."}), 201
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while registering the class.", "details": str(e)}), 500
    

@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        # Step 1: Get the image, folder name, and student ID from the request
        if 'image' not in request.files or 'folder_name' not in request.form or 'id' not in request.form:
            return jsonify({"error": "Image file, folder_name, and id are required"}), 400

        image_file = request.files['image']
        folder_name = request.form['folder_name']
        student_id = request.form['id']

        # Step 2: Check if the folder exists in the database
        class_folder_path = os.path.join('database', folder_name)

        if not os.path.exists(class_folder_path):
            return jsonify({"error": f"Class '{folder_name}' not found in the database."}), 404

        # Step 3: Create a subfolder for the student ID inside the class folder
        student_folder_path = os.path.join(class_folder_path, student_id)
        os.makedirs(student_folder_path, exist_ok=True)  # Create the student folder if it doesn't exist

        # Step 4: Save the uploaded image with the student ID as the filename inside the student folder
        timestamp_ms = int(time.time() * 1000) 
        image_path = os.path.join(student_folder_path, f"{timestamp_ms}.jpg")
        image_file.save(image_path)

        return jsonify({"message": f"Student image successfully registered with ID '{student_id}' in the class '{folder_name}'."}), 201

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while registering the student.", "details": str(e)}), 500

@app.route('/download_image/<path:filename>', methods=['GET'])
def download_image(filename):
    try:
        # Serve the file from the processed_images folder
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Unable to download image: {str(e)}"}), 500







if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
