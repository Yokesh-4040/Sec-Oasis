#!/usr/bin/env python3
"""
Multi-Image Training Setup for S-Oasis Face Recognition
======================================================

This script helps you set up multiple training images per person for better
face recognition accuracy. Instead of just one image per person, you can
provide multiple photos to train the system more effectively.

Usage:
    python3 create_training_images.py

The script will:
1. Show you current people in the database
2. Allow you to add additional training images for existing people
3. Provide tips for optimal face recognition training

Tips for best results:
- Use 3-5 different photos per person
- Include different angles, lighting conditions, and expressions
- Ensure faces are clearly visible and not obstructed
- Use high-quality images when possible
"""

import os
import shutil
from datetime import datetime

def get_dataset_people():
    """Get list of people currently in the dataset"""
    dataset_dir = "dataset"
    people = {}
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"[INFO] Created dataset directory: {dataset_dir}")
        return people
    
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(filename)[0]
            if '_' in base_name and base_name.split('_')[-1].isdigit():
                person_name = '_'.join(base_name.split('_')[:-1])
            else:
                person_name = base_name
            
            if person_name not in people:
                people[person_name] = []
            people[person_name].append(filename)
    
    return people

def add_training_image(person_name, source_image_path):
    """Add a new training image for an existing person"""
    dataset_dir = "dataset"
    
    if not os.path.exists(source_image_path):
        print(f"[ERROR] Source image not found: {source_image_path}")
        return False
    
    # Get current count for this person
    existing_count = 0
    for filename in os.listdir(dataset_dir):
        if filename.startswith(f"{person_name}_") or filename == f"{person_name}.jpg":
            existing_count += 1
    
    # Determine file extension
    _, ext = os.path.splitext(source_image_path)
    if not ext:
        ext = '.jpg'
    
    # Create new filename
    if existing_count == 0:
        new_filename = f"{person_name}{ext}"
    else:
        new_filename = f"{person_name}_{existing_count + 1}{ext}"
    
    new_path = os.path.join(dataset_dir, new_filename)
    
    try:
        shutil.copy2(source_image_path, new_path)
        print(f"[SUCCESS] Added training image: {new_filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy image: {e}")
        return False

def main():
    print("=" * 60)
    print("S-Oasis Multi-Image Training Setup")
    print("=" * 60)
    print()
    
    # Show current people in dataset
    people = get_dataset_people()
    
    if people:
        print("Current people in database:")
        print("-" * 30)
        for person, images in people.items():
            print(f"• {person}: {len(images)} image(s)")
            for img in images:
                print(f"  - {img}")
        print()
    else:
        print("No people found in database yet.")
        print("Add some initial images to the dataset/ directory first.")
        print()
        return
    
    print("Multi-Image Training Benefits:")
    print("• Better recognition accuracy")
    print("• Reduced false positives/negatives") 
    print("• More robust to lighting and angle changes")
    print("• Improved confidence scores")
    print()
    
    print("Training Tips:")
    print("• Use 3-5 different photos per person")
    print("• Include various angles and expressions")
    print("• Ensure good lighting and image quality")
    print("• Avoid blurry or obstructed faces")
    print()
    
    while True:
        print("Options:")
        print("1. Add training image for existing person")
        print("2. Show current database")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '1':
            if not people:
                print("No people in database to add images for.")
                continue
                
            print("\nAvailable people:")
            for i, person in enumerate(people.keys(), 1):
                print(f"{i}. {person}")
            
            try:
                person_choice = int(input("\nSelect person number: ")) - 1
                person_names = list(people.keys())
                
                if 0 <= person_choice < len(person_names):
                    selected_person = person_names[person_choice]
                    image_path = input(f"\nEnter path to new image for {selected_person}: ").strip()
                    
                    if add_training_image(selected_person, image_path):
                        people = get_dataset_people()  # Refresh the list
                        print(f"\n[INFO] {selected_person} now has {len(people[selected_person])} training images")
                    
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == '2':
            people = get_dataset_people()  # Refresh
            if people:
                print("\nCurrent database:")
                for person, images in people.items():
                    print(f"• {person}: {len(images)} image(s)")
            else:
                print("No people in database.")
                
        elif choice == '3':
            print("\nExiting. Remember to restart S-Oasis to load new training images!")
            break
        else:
            print("Invalid option. Please choose 1, 2, or 3.")
        
        print()

if __name__ == "__main__":
    main() 