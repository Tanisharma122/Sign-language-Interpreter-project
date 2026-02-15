import streamlit as st
import os
import threading
import http.server
import socketserver

# Set page config
st.set_page_config(
    page_title="SignLink Pro - Avatar",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration for static server
PORT = 8501  # Standard Streamlit port is 8501
ASSET_PORT = 8005
DIRECTORY = "static"
detector = None

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            # Initialize detector if needed (singleton pattern or per request? Per request blocks server if not threaded properly for other requests, but this is simple demo)
            # Better to use a global detector for this single-user demo
            global detector
            if detector is None:
                try:
                    # Adjust path to find mediapipe folder
                    import sys
                    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                    from mediapipe.sign_detector import SignLanguageDetector
                    # Paths need to be absolute or relative to where script is run
                    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mediapipe'))
                    model_path = os.path.join(root_dir, 'sign_language_lstm.pth')
                    labels_path = os.path.join(root_dir, 'labels.pkl')
                    detector = SignLanguageDetector(model_path, labels_path)
                except Exception as e:
                    print(f"Error initializing detector: {e}")
                    return

            try:
                for frame in detector.generate_frames():
                    self.wfile.write(frame)
            except Exception as e:
                print(f"Stream closed: {e}")
                detector.stop_camera()

        elif self.path == '/run_mediapipe':
             # Deprecated or kept for compatibility, but we prefer video_feed
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Use /video_feed instead")
        else:
            super().do_GET()

def start_server():
    """Starts a simple HTTP server to serve static files (HTML, GLB, CSS, JS)"""
    try:
        with socketserver.TCPServer(("", ASSET_PORT), Handler) as httpd:
            print(f"Serving static assets at port {ASSET_PORT}")
            httpd.serve_forever()
    except OSError:
        print(f"Port {ASSET_PORT} might already be in use. Assuming server is running.")

# Start the static file server in a background thread
if "server_started" not in st.session_state:
    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()
    st.session_state.server_started = True

# Embed the application using an iframe
# This ensures that relative paths for GLB files work correctly within the HTML
st.components.v1.iframe(f"http://localhost:{ASSET_PORT}/index.html", height=1000, scrolling=True)
