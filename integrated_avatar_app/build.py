
import os

def build():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, "..", "index.html")
    dest_path = os.path.join(base_dir, "static", "index.html")

    with open(source_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Inject Head Dependencies
    head_injection = """
    <!-- Three.js Dependencies -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    """
    content = content.replace('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', 
                              '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">\n' + head_injection)

    # 2. Inject CSS
    css_injection = """
        /* Avatar Integration Styles */
        #avatar-container {
            width: 100%;
            height: 400px;
            position: relative;
            background: radial-gradient(circle at center, #2a2a40 0%, #000 100%);
            border-radius: var(--radius-lg);
            overflow: hidden;
            margin-bottom: 24px;
            border: 1px solid var(--primary);
        }
        #viewer {
            width: 100%;
            height: 100%;
            display: block;
        }
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            background: rgba(0,0,0,0.6);
            z-index: 10;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        .loading-overlay.active {
            opacity: 1;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    """
    content = content.replace('</style>', css_injection + '\n    </style>')

    # 3. Replace Avatar Container
    target_div = '<div class="sign-avatar" id="signAvatar">\n                        <i class="fas fa-hand-peace"></i>\n                    </div>'
    new_div = """
                    <div id="avatar-container">
                        <canvas id="viewer"></canvas>
                        <div id="loading" class="loading-overlay">
                            <div class="spinner"></div>
                        </div>
                    </div>
    """
    # Try exact match, if fails, be more lenient or use regex.
    # The read content might have different whitespace.
    # Let's find the location of id="signAvatar" and replace the whole div.
    if target_div not in content:
        # Fallback: simple replacement using larger context if needed, or just find by ID
        import re
        content = re.sub(r'<div class="sign-avatar" id="signAvatar">[\s\S]*?</div>', new_div, content)
    else:
        content = content.replace(target_div, new_div)


    # 4. Inject JS Logic
    js_logic = """
    <script>
        // ==================== AVATAR LOGIC ====================
        let scene, camera, renderer, controls, mixer;
        let clock = new THREE.Clock();
        let currentModel = null;
        let viewer, loader;
        let animationQueue = [];
        let isProcessingQueue = false;

        // Word Map
        const WORD_MAP = {
            'hello': 'Waving.glb', 'hi': 'Waving.glb', 'hey': 'Waving.glb', 'wave': 'Waving.glb',
            'waving': 'Waving.glb', 'greet': 'Standing Greeting.glb', 'greeting': 'Standing Greeting.glb',
            'salute': 'Salute.glb', 'welcome': 'Standing Greeting.glb',
            'handshake': 'Shaking Hands 1.glb', 'shake': 'Shaking Hands 1.glb',
            'thanks': 'Thankful.glb', 'thank': 'Thankful.glb', 'thank you': 'Thankful.glb',
            'praying': 'Praying.glb', 'pray': 'Praying.glb', 'please': 'Praying.glb', 'namaste': 'Praying.glb',
            'bow': 'Quick Formal Bow.glb', 'respect': 'Quick Formal Bow.glb',
            'clap': 'Standing Clap.glb', 'clapping': 'Standing Clap.glb', 'bravo': 'Standing Clap.glb',
            'good': 'Standing Thumbs Up.glb', 'thumbs up': 'Standing Thumbs Up.glb', 'like': 'Standing Thumbs Up.glb',
            'yes': 'Standing Thumbs Up.glb', 'approve': 'Standing Thumbs Up.glb',
            'pointing': 'Pointing.glb', 'point': 'Pointing.glb', 'look': 'Pointing.glb',
            'counting': 'Counting.glb', 'count': 'Counting.glb', 'numbers': 'Counting.glb', 'one': 'Counting.glb',
            'idle': 'avatar1.glb', 'stop': 'avatar1.glb'
        };

        function initAvatar() {
            viewer = document.getElementById('viewer');
            if (!viewer) return;

            loader = new THREE.GLTFLoader();
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1e293b);

            const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
            hemiLight.position.set(0, 20, 0);
            scene.add(hemiLight);
            const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
            dirLight.position.set(5, 10, 7);
            scene.add(dirLight);

            camera = new THREE.PerspectiveCamera(45, viewer.clientWidth / viewer.clientHeight, 0.1, 100);
            camera.position.set(0, 1.5, 3.5);

            renderer = new THREE.WebGLRenderer({ canvas: viewer, antialias: true, alpha: true });
            renderer.setSize(viewer.clientWidth, viewer.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.outputEncoding = THREE.sRGBEncoding;

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(0, 1, 0);
            controls.enableDamping = true;
            controls.minDistance = 2;
            controls.maxDistance = 6;

            window.addEventListener('resize', onWindowResize);
            animate();
            loadAnimation('avatar1.glb', 'Idle');
        }

        function onWindowResize() {
            if (!viewer) return;
            const container = viewer.parentElement;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            const delta = clock.getDelta();
            if (mixer) mixer.update(delta);
            controls.update();
            renderer.render(scene, camera);
        }

        // Modified loadAnimation to support sequential playback with callback support
        function loadAnimation(filename, actionName, cb, isSequence) {
            const loadingEl = document.getElementById('loading');
            
            // Only show loader if not in a fast sequence (optional, to avoid flicker)
            // But for now let's show it only if not sequence or if it takes time
           
            loader.load(filename, (gltf) => {
                if (currentModel) scene.remove(currentModel);
                currentModel = gltf.scene;
                scene.add(currentModel);

                mixer = new THREE.AnimationMixer(currentModel);
                let duration = 0;

                if (gltf.animations.length > 0) {
                    const action = mixer.clipAction(gltf.animations[0]);
                    action.setEffectiveTimeScale(1.2); // Speed up slightly
                    action.setLoop(THREE.LoopOnce); // Play once!
                    action.clampWhenFinished = true; // Stay at end frame
                    action.play();
                    duration = gltf.animations[0].duration;
                }

                // If this is part of a sequence, we need to handle the duration
                if (isSequence && cb) {
                     // Wait for animation to finish (approx) then call callback
                     // Note: duration is in seconds.
                     const ms = (duration / 1.2) * 1000; 
                     setTimeout(cb, ms + 500); // Small buffer
                } else if (cb) {
                    cb();
                }

            }, undefined, (e) => {
                console.error(e);
                if (cb) cb(); // proceed in case of error
            });
        }

        function speak(text) {
             if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                speechSynthesis.speak(utterance);
            }
        }

        // ==================== QUEUE SYSTEM ====================
        
        window.translateVoice = function() {
            const text = document.getElementById('voiceInput').value;
            if (!text) return;
            const signText = document.getElementById('signText');
            
            signText.textContent = "Processing sequence...";
            animationQueue = [];

            // Tokenize
            const words = text.toLowerCase().split(/\\s+/);
            
            words.forEach(word => {
                const clean = word.replace(/[^a-z]/g, '');
                if (WORD_MAP[clean]) {
                     animationQueue.push({ type: 'action', file: WORD_MAP[clean], name: clean });
                } else {
                     // For non-sign words, maybe just speak them?
                     // User said "sync with words properly... hello thanks clap if they all are together"
                     // This implies we should do them one by one.
                     // Let's treat them as just speech items.
                     animationQueue.push({ type: 'speech', text: word });
                }
            });

            // Start Queue
            processQueue();
        };

        function processQueue() {
            if (animationQueue.length === 0) {
                // Done
                setTimeout(() => loadAnimation('avatar1.glb', 'Idle'), 1000);
                document.getElementById('signText').textContent = "Sequence finished.";
                return;
            }

            const item = animationQueue.shift();
            const signText = document.getElementById('signText');

            if (item.type === 'action') {
                signText.textContent = `Signing: "${item.name}"`;
                speak(item.name); // Speak the word while signing
                loadAnimation(item.file, item.name, () => {
                    processQueue(); // Next after animation ends
                }, true);
            } else {
                signText.textContent = `Speaking: "${item.text}"`;
                speak(item.text);
                // Heuristic delay for speech
                const delay = Math.max(1000, item.text.length * 100); 
                setTimeout(processQueue, delay);
            }
        }

        const originalStartVoiceToSign = window.startVoiceToSign;
        window.startVoiceToSign = function() {
            originalStartVoiceToSign();
            setTimeout(initAvatar, 100);
        };

        // Override toggleCamera to launch MediaPipe app instead of browser webcam
        window.toggleCamera = function() {
            const btn = document.getElementById('cameraBtn');
            const placeholder = document.getElementById('videoPlaceholder');
            const video = document.getElementById('webcam');
            
            // Trigger backend script
            fetch('/run_mediapipe')
                .then(response => {
                    if (response.ok) {
                        btn.innerHTML = '<i class="fas fa-check"></i> Launched';
                        document.getElementById('output').textContent = "MediaPipe Live Demo Launched!";
                        setTimeout(() => { btn.innerHTML = '<i class="fas fa-camera"></i> Start Camera'; }, 3000);
                    } else {
                        alert('Failed to launch script');
                    }
                })
                .catch(err => {
                    console.error(err);
                    alert('Error reaching backend');
                });
        };
    </script>
    """
    content = content.replace('</body>', js_logic + '\n</body>')

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    build()
