#!/usr/bin/env python3
"""
Cotton Disease Prediction Application Startup Script
"""

import os
import sys
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dataset():
    """Check if the dataset exists and has the correct structure"""
    dataset_path = "cotton"
    if not os.path.exists(dataset_path):
        print("❌ Error: Dataset folder 'cotton' not found!")
        print("Please ensure your dataset is in the 'cotton' folder with the following structure:")
        print("cotton/")
        print("├── bacterial_blight/")
        print("├── curl_virus/")
        print("├── fussarium_wilt/")
        print("└── healthy/")
        return False
    
    required_classes = ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy']
    missing_classes = []
    
    for class_name in required_classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
        else:
            # Count images in the class
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
            print(f"✅ {class_name}: {image_count} images")
    
    if missing_classes:
        print(f"❌ Missing classes: {', '.join(missing_classes)}")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please run manually:")
        print("pip install -r requirements.txt")
        return False

def main():
    """Main startup function"""
    print("🌿 Cotton Disease Prediction Application")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dataset
    print("\n📁 Checking dataset...")
    if not check_dataset():
        return
    
    # Check if requirements are installed
    try:
        import flask
        import tensorflow
        print("✅ Required packages are already installed!")
    except ImportError:
        print("📦 Some required packages are missing.")
        install_choice = input("Would you like to install them now? (y/n): ").lower()
        if install_choice == 'y':
            if not install_requirements():
                return
        else:
            print("Please install requirements manually: pip install -r requirements.txt")
            return
    
    # Start the application
    print("\n🚀 Starting the application...")
    print("The web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main() 