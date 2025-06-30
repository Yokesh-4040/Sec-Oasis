"""
Simple RTSP Connection Test for Tapo C520 Camera
Use this script to verify your camera connection before running the full detection system
"""

import cv2
import os
import time
from camera_config import get_rtsp_url, print_config

def test_rtsp_connection():
    """Test RTSP connection to Tapo C520 camera"""
    
    print_config()
    print("\n[INFO] Testing RTSP connection...")
    
    rtsp_url = get_rtsp_url()
    
    # Set OpenCV capture options for better RTSP performance
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    
    # Try to connect to camera
    print(f"[INFO] Attempting to connect to camera...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open RTSP stream!")
        print(f"[ERROR] Please check:")
        print(f"         1. Camera IP address is correct")
        print(f"         2. Camera is powered on and connected to network")
        print(f"         3. Username and password are correct")
        print(f"         4. Camera account is created in Tapo app")
        print(f"         5. Camera is accessible from this computer")
        return False
    
    print(f"[SUCCESS] Connected to camera!")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Stream properties:")
    print(f"        Resolution: {width}x{height}")
    print(f"        FPS: {fps}")
    
    # Test frame capture
    frame_count = 0
    start_time = time.time()
    successful_frames = 0
    
    print(f"[INFO] Testing frame capture for 10 seconds...")
    print(f"[INFO] Press 'q' to quit early, 's' to save test image")
    
    while time.time() - start_time < 10:  # Test for 10 seconds
        ret, frame = cap.read()
        
        if ret:
            successful_frames += 1
            frame_count += 1
            
            # Add timestamp overlay
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Test Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, timestamp, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "RTSP Connection Test", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('RTSP Connection Test', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] User requested quit")
                break
            elif key == ord('s'):
                # Save test image
                test_filename = f"rtsp_test_{int(time.time())}.jpg"
                cv2.imwrite(test_filename, frame)
                print(f"[INFO] Test image saved: {test_filename}")
        else:
            print(f"[WARNING] Failed to read frame {frame_count + 1}")
        
        frame_count += 1
    
    # Calculate statistics
    test_duration = time.time() - start_time
    success_rate = (successful_frames / frame_count) * 100 if frame_count > 0 else 0
    actual_fps = successful_frames / test_duration
    
    print(f"\n[INFO] Connection test results:")
    print(f"        Test duration: {test_duration:.1f} seconds")
    print(f"        Total frames attempted: {frame_count}")
    print(f"        Successful frames: {successful_frames}")
    print(f"        Success rate: {success_rate:.1f}%")
    print(f"        Actual FPS: {actual_fps:.1f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Provide feedback
    if success_rate >= 90:
        print(f"[SUCCESS] Connection test passed! Stream is stable.")
        print(f"[INFO] You can now run the full detection system.")
        return True
    elif success_rate >= 70:
        print(f"[WARNING] Connection is somewhat unstable ({success_rate:.1f}% success rate)")
        print(f"[SUGGESTION] Consider:")
        print(f"            - Moving camera closer to router")
        print(f"            - Using wired connection instead of WiFi")
        print(f"            - Reducing stream quality (use stream2 instead of stream1)")
        return True
    else:
        print(f"[ERROR] Connection test failed! ({success_rate:.1f}% success rate)")
        print(f"[SUGGESTION] Check network connectivity and camera settings")
        return False

def test_network_connectivity():
    """Test basic network connectivity to camera"""
    from camera_config import CAMERA_CONFIG
    import subprocess
    import platform
    
    camera_ip = CAMERA_CONFIG['ip']
    print(f"[INFO] Testing network connectivity to {camera_ip}...")
    
    # Ping test
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "3", camera_ip]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[SUCCESS] Camera is reachable via ping")
            return True
        else:
            print(f"[ERROR] Camera is not reachable via ping")
            print(f"[SUGGESTION] Check if camera IP address is correct: {camera_ip}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Ping test timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Network test failed: {e}")
        return False

def main():
    """Main test function"""
    print("TP-Link Tapo C520 RTSP Connection Test")
    print("=" * 50)
    
    # Test network connectivity first
    if not test_network_connectivity():
        print(f"\n[ERROR] Basic network connectivity failed!")
        print(f"[SUGGESTION] Verify camera IP address and network connection")
        return
    
    print()
    
    # Test RTSP connection
    if test_rtsp_connection():
        print(f"\n[SUCCESS] All tests passed!")
        print(f"[INFO] Your camera is ready for live PPE detection")
        print(f"[INFO] Run 'python detect_ppe_live_stream.py' to start detection")
    else:
        print(f"\n[ERROR] RTSP connection test failed!")
        print(f"[SUGGESTION] Please check camera configuration and try again")

if __name__ == "__main__":
    main() 