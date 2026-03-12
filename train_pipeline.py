import os
import numpy as np
import cv2
from deepface import DeepFace
import joblib
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shutil
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

def train_face_recognition_pipeline(data_root_path, model_save_path = "saved_models", random_state=42):
    """
    Complete training pipeline for face recognition using KNN
    Generates 10 augmented images per original image
    
    Parameters:
    -----------
    data_root_path : str
        Path to the root directory containing person folders
        Each person folder should contain 15 images (5 from 3 angles)
    
    model_save_path : str
        Directory path where trained models will be saved
    
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing training results
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs("cropped_dataset", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    
    # Step 1: Data loading and face extraction
    print("Step 1: Loading images and extracting faces...")
    X = []  # Features (face images)
    y = []  # Labels (person names)
    person_names = []
    
    # Get all person folders
    person_folders = [f for f in os.listdir(data_root_path) 
                     if os.path.isdir(os.path.join(data_root_path, f))]
    
    for label_idx, person_name in enumerate(tqdm(person_folders)):
        person_path = os.path.join(data_root_path, person_name)
        
        # Get all images for this person
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Verify each person has exactly 15 images
        # if len(image_files) != 15:
        #     print(f"Warning: Person '{person_name}' has {len(image_files)} images, expected 15")
        
        os.makedirs(f"cropped_dataset/{person_path}", exist_ok=True)

        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            
            try:
                # Extract face using DeepFace
                # DeepFace returns face image of size 224x224, we'll resize to 128x128
                face_img = DeepFace.extract_faces(image_path, detector_backend="ssd", align=False, normalize_face=False)
                
                # Convert from RGB (DeepFace returns RGB) and scale back to 0-255
                # face_img = (face_img * 255).astype(np.uint8)
                
                face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                face_img_gray = cv2.resize(face_img_gray, (128, 128), cv2.INTER_AREA)

                cv2.imwrite(f"cropped_dataset/{person_path}", face_img_gray)

            except Exception as e:
                print(f"Warning: Could not process {image_path}: {str(e)}")
                continue
    
    if len(X) == 0:
        raise ValueError("No faces were extracted from the provided images")
    
    person_names = list(person_folders)
    for face_folder in person_names:
        images_list = os.listdir(os.path.join("cropped_dataset", face_folder))

        train_path = os.path.join("train", face_folder)
        test_path = os.path.join("test", face_folder)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        train_set, test_set = train_test_split(images_list, test_size=0.2, random_state=42)

        for img in train_set:
            
            src = os.path.join("cropped_dataset", face_folder, img)
            dest = os.path.join(train_path, img)

            shutil.copy(src, dest)

        for img in test_set:

            src = os.path.join("cropped_dataset", face_folder, img)
            dest = os.path.join(test_path, img)

            shutil.copy(src, dest)

        

    
    # original_samples_per_person = len(X) // len(person_names)
    # print(f"Original samples extracted: {len(X)}")
    # print(f"  - Number of classes: {len(person_names)}")
    # print(f"  - Original samples per person: {original_samples_per_person}")
    
    # # Step 2: Data augmentation - Generate exactly 10 augmented images per original image
    # print("\nStep 2: Generating 10 augmented images per original image...")
    
    # Define augmentation pipeline with random combinations
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.GaussNoise(std_range=[0.05, 0.1], p=0.2),
    ])
    
    # Apply augmentations to generate exactly 10 per original image
    # X_augmented = []
    # y_augmented = []
    
    # for img_idx, (img, label) in enumerate(tqdm(zip(X, y), total=len(X), desc="Generating 10 augmentations per image")):
    #     # Add original image
    #     X_augmented.append(img)
    #     y_augmented.append(label)
        
    #     # Generate exactly 10 augmented versions
    #     for aug_num in range(10):
    #         # Set different random seed for each augmentation to ensure variety
    #         np.random.seed(random_state + img_idx * 100 + aug_num)
            
    #         try:
    #             augmented = augmentation_pipeline(image=img)['image']
    #             X_augmented.append(augmented)
    #             y_augmented.append(label)
    #         except Exception as e:
    #             print(f"Warning: Augmentation {aug_num+1} failed for image {img_idx}: {str(e)}")
    #             # If augmentation fails, add the original image as fallback
    #             X_augmented.append(img)
    #             y_augmented.append(label)
    
    # X_augmented = np.array(X_augmented)
    # y_augmented = np.array(y_augmented)

    augmentations = 10

    for face_folder in os.listdir("train"):
        print(f"Reading images from {face_folder}")
        image_list = os.listdir(os.path.join("train", face_folder))
        for image in image_list:
            # plt.figure(figsize=(2, 2))
            img = cv2.imread(os.path.join(f"train/{face_folder}/{image}"), cv2.IMREAD_GRAYSCALE)
            # print(f"Shape of input image is {img.shape}")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_path = os.path.join("train", face_folder)
            
            for i in range(augmentations):
                augmented = augmentation_pipeline(image = img)
                augm_img = augmented["image"]
                cv2.imwrite(f"{save_path}/{image}_aug_{i}.jpg", augm_img)
    
    # # Calculate samples per person
    # samples_per_person = len(X_augmented) // len(person_names)
    
    # print(f"\nAugmentation Results:")
    # print(f"  - Original samples: {len(X)}")
    # print(f"  - Total samples after augmentation: {len(X_augmented)}")
    # print(f"  - Samples per person: {samples_per_person}")
    # print(f"  - Augmentation factor: {len(X_augmented)/len(X):.1f}x")
    # print(f"  - Expected (15 original × 11 = 165): {samples_per_person}")
    
    # # Verify we have 165 samples per person (15 original + 150 augmented)
    # expected_per_person = 165  # 15 original + (15 × 10 augmented)
    # if samples_per_person != expected_per_person:
    #     print(f"Warning: Expected {expected_per_person} samples per person, got {samples_per_person}")
    
    # Step 3: Flatten and normalize images
    print("\nStep 3: Flattening and normalizing images...")
    
    train_data = []
    train_labels = []
    data_path = "train"
    classes = os.listdir(data_path)
    for face_folder in classes:
        for face in os.listdir(os.path.join(data_path, face_folder)):
            image = cv2.imread(os.path.join(data_path, f"{face_folder}/{face}"), cv2.IMREAD_GRAYSCALE)
            
            image_flattened = image.flatten() / 255.0
            train_data.append(image_flattened)
            train_labels.append(classes.index(face_folder))

    print(f"person folders: {person_folders}")
    print(f"class names: {train_labels}")

    test_data = []
    test_labels = []
    data_path = "test"
    classes = os.listdir(data_path)
    for face_folder in classes:
        for face in os.listdir(os.path.join(data_path, face_folder)):
            image = cv2.imread(os.path.join(data_path, f"{face_folder}/{face}"), cv2.IMREAD_GRAYSCALE)
            
            image_flattened = image.flatten() / 255.0
            test_data.append(image_flattened)
            test_labels.append(classes.index(face_folder))
    
    train_df = pd.DataFrame(train_data)
    train_df["label"] = train_labels

    test_df = pd.DataFrame(test_data)
    test_df["label"] = test_labels

    x_train = train_df.drop("label", axis = 1).values

    y_train = train_df["label"].values

    x_test = test_df.drop("label", axis = 1).values

    y_test = test_df["label"].values



    # Step 4: Apply PCA
    print("\nStep 4: Applying PCA to reduce dimensionality to 280 features...")
    
    # Initialize and fit PCA on all data
    pca = PCA(n_components=0.95, random_state=random_state)
    X_pca = pca.fit_transform(x_train)
    
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")
    print(f"Data after PCA: {X_pca.shape}")
    
    # Step 5: Train KNN model
    print("\nStep 5: Training KNN classifier with Euclidean distance...")
    
    # Calculate optimal k based on samples per person
    recommended_k = 2 # Avoid k being too large
    
    knn = KNeighborsClassifier(
        n_neighbors=recommended_k,
        metric='euclidean',
        weights='distance',
        algorithm='auto',
        n_jobs=-1  # Use all available CPU cores
    )
    
    knn.fit(X_pca, y_train)
    
    # Step 6: Save models
    print("\nStep 6: Saving models...")
    
    # Save PCA model
    pca_save_path = os.path.join(model_save_path, 'pca_model.pkl')
    joblib.dump(pca, pca_save_path)
    
    # Save KNN model
    knn_save_path = os.path.join(model_save_path, 'knn_model.pkl')
    joblib.dump(knn, knn_save_path)

    x_test_pca = pca.transform(x_test)

    y_pred_final = knn.predict(x_test_pca)

    conf_mat = confusion_matrix(y_test, y_pred_final)

    accuracy = accuracy_score(y_test, y_pred_final)

    plt.figure(figsize=(10,8))

    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(f"KNN Confusion Matrix\nAccuracy = {accuracy:.4f}")

    plt.savefig("knn_confusion_matrix.png", bbox_inches="tight")

    plt.close()
            
    print("New KNN model trained and saved.")
    # # Save training info for reference
    # training_info = {
    #     'original_samples': len(X),
    #     'total_samples': len(X_augmented),
    #     'num_classes': len(person_names),
    #     'samples_per_class': samples_per_person,
    #     'augmentation_factor': len(X_augmented)/len(X),
    #     'expected_per_class': 165,
    #     'pca_components': n_components,
    #     'pca_explained_variance': float(sum(pca.explained_variance_ratio_)),
    #     'knn_k': recommended_k,
    #     'person_names': person_names,
    #     'augmentation_config': {
    #         'augmentations_per_image': 10,
    #         'pipeline': 'HorizontalFlip, Rotate, RandomBrightnessContrast, GaussianBlur, GaussNoise'
    #     }
    # }
    
    # info_path = os.path.join(model_save_path, 'training_info.pkl')
    # with open(info_path, 'wb') as f:
    #     pickle.dump(training_info, f)
    
    # print(f"Models saved to: {model_save_path}")
    # print(f"  - PCA model: {pca_save_path}")
    # print(f"  - KNN model: {knn_save_path}")
    # print(f"  - Training info: {info_path}")
    
    # print("\n" + "="*50)
    # print("TRAINING COMPLETED SUCCESSFULLY!")
    # print("="*50)
    # print(f"Final dataset: {len(X_augmented)} images")
    # print(f"  - {len(person_names)} persons")
    # print(f"  - {samples_per_person} images per person")
    # print(f"  - {samples_per_person - 15} augmented images per person")
    # print("="*50)
    
    # return {
    #     'status': 'success',
    #     'message': f'Model trained with {len(X_augmented)} images ({samples_per_person} per person)',
    #     'num_classes': len(person_names),
    #     'original_samples': int(len(X)),
    #     'total_samples': int(len(X_augmented)),
    #     'samples_per_person': int(samples_per_person),
    #     'augmented_per_original': 10,
    #     'pca_explained_variance': float(sum(pca.explained_variance_ratio_)),
    #     'knn_neighbors': int(recommended_k),
    #     'model_paths': {
    #         'pca': pca_save_path,
    #         'knn': knn_save_path,
    #         'info': info_path
    #     }
    # }

