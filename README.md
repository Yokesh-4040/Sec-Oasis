# PPE Detection System with Web Dashboard

A real-time Personal Protective Equipment (PPE) detection system using AI and computer vision. Features a modern React-based web dashboard for monitoring workplace safety compliance.

## Features

🎥 **Live Camera Feed** - Real-time monitoring via TP-Link Tapo C520 camera  
🔍 **AI Detection** - YOLO-based PPE detection (Hardhat, Mask, Safety Vest)  
👤 **Face Recognition** - Employee identification system  
📊 **Web Dashboard** - Modern React interface with real-time updates  
📝 **Compliance Logging** - Automated CSV logs with timestamps  
⚡ **WebSocket Updates** - Live data streaming to dashboard  
📱 **Responsive Design** - Works on desktop, tablet, and mobile  

## System Requirements

- Python 3.8+
- macOS, Windows, or Linux
- TP-Link Tapo C520 camera (or compatible RTSP camera)
- 4GB+ RAM (for face recognition processing)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sec-Oasis

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Camera Setup

1. Configure your TP-Link Tapo camera:
   - Enable RTSP in Tapo app
   - Create camera account (Advanced Settings → Camera Account)
   - Note down IP address

2. Update camera configuration in `camera_config.py`:
   ```python
   CAMERA_CONFIG = {
       "ip": "YOUR_CAMERA_IP",
       "username": "YOUR_USERNAME", 
       "password": "YOUR_PASSWORD"
   }
   ```

### 3. Add Employee Faces

Place employee photos in the `dataset/` directory:
```
dataset/
├── John_Doe.jpg
├── Jane_Smith.jpg
└── ... (more employee photos)
```

### 4. Start the System

```bash
python3 app.py
```

Open your browser to: **http://localhost:8080**

## Web Dashboard Usage

### Main Interface
- **Live Feed**: Real-time camera stream with detection overlays
- **Detection Panel**: Lists detected people and PPE items
- **System Stats**: FPS, frames processed, connection status
- **Compliance Status**: Real-time safety compliance monitoring

### Controls
- **🟢 START**: Begin live detection
- **🔴 STOP**: Stop the system
- **🔄 Refresh**: Update logs and status

### Detection Results
- **Green boxes**: Required PPE items (compliant)
- **Red boxes**: Other detected items
- **Purple boxes**: Detected faces
- **Status display**: COMPLIANT/NON-COMPLIANT

## File Structure

```
Sec-Oasis/
├── app.py                    # Main Flask web application
├── camera_config.py          # Camera and system configuration
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # React-based dashboard
├── dataset/                 # Employee face images
├── results/                 # Detection logs and snapshots
├── test_images/            # Static test images
├── best.pt                 # YOLO model weights
├── detect_ppe_and_names.py # Static image detection
├── test_rtsp_connection.py # Camera diagnostics
└── setup_system.py        # Automated setup utility
```

## Configuration Options

### Camera Settings (`camera_config.py`)
```python
CAMERA_CONFIG = {
    "ip": "192.168.0.245",           # Camera IP address
    "username": "Admin_FF",          # RTSP username
    "password": "Fourty40!",         # RTSP password
    "stream_path": "stream1",        # HD stream (stream2 for SD)
    "fps_limit": 15,                 # Frame rate limit
    "frame_skip": 3                  # Process every 3rd frame
}
```

### Detection Settings
```python
DETECTION_CONFIG = {
    "required_ppe": ["Hardhat", "Mask", "Safety Vest"],
    "yolo_confidence": 0.5,          # Detection confidence threshold
    "face_tolerance": 0.6,           # Face recognition sensitivity
    "log_interval": 30               # Log every 30 detections
}
```

## API Endpoints

The system provides REST APIs for integration:

- `GET /api/status` - System status and current detections
- `POST /api/start` - Start detection system
- `POST /api/stop` - Stop detection system
- `GET /api/config` - System configuration
- `GET /api/logs` - Recent detection logs

WebSocket endpoint: `/socket.io` for real-time updates

## Troubleshooting

### Camera Connection Issues
```bash
# Test RTSP connection
python3 test_rtsp_connection.py
```

### Common Solutions
- **Port 8080 in use**: Change port in `app.py`
- **Camera not found**: Verify IP address and credentials
- **No face detection**: Ensure proper lighting and image quality
- **Slow performance**: Reduce FPS or increase frame skip

## Static Image Detection

For testing with static images:
```bash
python3 detect_ppe_and_names.py
```

Results will be saved in the `results/` directory.

## Dependencies

- **Flask**: Web framework and API server
- **OpenCV**: Computer vision and video processing
- **YOLO (Ultralytics)**: Object detection model
- **face_recognition**: Face detection and recognition
- **Flask-SocketIO**: Real-time web communication
- **React**: Frontend dashboard (loaded via CDN)

## Security Notes

- Change default camera credentials
- Use HTTPS in production
- Implement user authentication for dashboard access
- Secure network access to camera streams

## Performance Optimization

- Adjust `frame_skip` to reduce processing load
- Lower `fps_limit` for slower hardware
- Use `stream2` for lower resolution (faster processing)
- Reduce YOLO confidence threshold for faster detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly with your camera setup
4. Submit a pull request

## License

This project is open source. See LICENSE file for details.

---

**Built with ❤️ for workplace safety**

For support or questions, please open an issue on GitHub. 