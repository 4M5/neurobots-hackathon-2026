/**
 * AI EXAM PROCTOR - Main Application
 * ===================================
 * Complete exam proctoring system with:
 * - Face detection & gaze tracking (MediaPipe)
 * - Keystroke analysis & behavior detection
 * - Tab switching & focus monitoring
 * - Clipboard analysis
 * - Audio monitoring
 * - Oral verification
 */

// ============================================
// GLOBAL STATE
// ============================================
const state = {
    student: { name: '', id: '' },
    exam: {
        currentQuestion: 0,
        answers: [],
        startTime: null,
        duration: 30 * 60, // 30 minutes in seconds
        timeRemaining: 30 * 60,
        sessionId: null,
        sessionToken: null,
        submissionStatus: 'not-started',
        verificationComplete: false,
        verificationSaveStatus: 'not-started'
    },
    proctoring: {
        warnings: 0,
        maxWarnings: 5,
        needsManualReview: false,
        tabSwitches: 0,
        focusLosses: 0,
        pasteEvents: 0,
        gazeAway: 0,
        typingBursts: 0,
        logs: []
    },
    typing: {
        lastKeyTime: 0,
        keyIntervals: [],
        examKeyIntervals: [],
        backspaceCount: 0,
        keystrokeCount: 0,
        editCount: 0,
        idleDurationsSeconds: [],
        lastInputTime: 0,
        burstThreshold: 50, // ms - keys faster than this are burst typing
        burstCount: 0
    },
    behaviorPrediction: { status: 'not-requested', result: null },
    audio: {
        context: null,
        analyser: null,
        isActive: false
    },
    face: {
        mesh: null,
        faceDetected: false,
        noFaceLogged: false,
        isLookingAway: false,
        lookAwayStart: null,
        gazeThreshold: 0.15
    },
    verification: {
        currentQuestion: 0,
        responses: []
    },
    mediaStream: null,
    cameraStream: null,
    microphoneStream: null,
    animationFrame: null,
    mediaCamera: null,
    isSetupComplete: false
};

// Exam questions container
let examQuestions = [];
const lifecycle = ExamLifecycle.createExamLifecycle({ onTimeout: () => {
    addLog('⏰ Time is up!', 'danger');
    endExam(false);
}, onTick: (seconds) => {
    state.exam.timeRemaining = seconds;
    document.getElementById('time-display').textContent = formatTime(seconds);
    if (seconds === 300) {
        document.getElementById('timer').classList.add('warning');
        addLog('⏰ 5 minutes remaining', 'warning');
    }
} });
let questionNavigation = null;
let verificationTimeout = null;
let recognitionActive = false;
let finalTranscript = '';
let startPending = false;

// Fallback questions (if API fails)
const fallbackQuestions = [
    {
        id: 1,
        text: "Explain the concept of Object-Oriented Programming (OOP) and list its four main principles with brief descriptions.",
        topic: "OOP Principles"
    },
    {
        id: 2,
        text: "What is the difference between a stack and a queue data structure? Provide examples of real-world applications for each.",
        topic: "Data Structures"
    },
    {
        id: 3,
        text: "Describe what an API (Application Programming Interface) is and explain the difference between REST and GraphQL APIs.",
        topic: "APIs"
    },
    {
        id: 4,
        text: "What is machine learning? Explain the difference between supervised and unsupervised learning with examples.",
        topic: "Machine Learning"
    },
    {
        id: 5,
        text: "Explain the concept of database normalization. What are the benefits and when might you choose to denormalize?",
        topic: "Databases"
    }
];

async function fetchQuestions() {
    try {
        // Use relative URL to work with Flask serving static files
        const response = await fetch('/api/questions');
        if (!response.ok) throw new Error('API request failed');
        
        const questions = await response.json();
        if (!Array.isArray(questions) || questions.length !== 5 || questions.some(question =>
            !question || typeof question.text !== 'string' || !question.text.trim() || question.text.length > 2000 ||
            typeof question.topic !== 'string' || !question.topic.trim() || question.topic.length > 100)) {
            throw new Error('Invalid question response');
        }
        examQuestions = questions.map((question, index) => ({
            id: index + 1,
            text: question.text.trim(),
            topic: question.topic.trim()
        }));
        console.log('Questions loaded from API');
    } catch (err) {
        console.error('Error fetching questions:', err);
        console.log('Using offline question set');
        examQuestions = [...fallbackQuestions];
    }
}

// Verification questions (based on exam answers)
const verificationQuestions = [
    "You mentioned OOP principles in your answer. Can you explain polymorphism in your own words?",
    "Can you describe a real-world scenario where you would use a queue over a stack?",
    "How would you explain API authentication to someone new to programming?"
];

// ============================================
// UTILITY FUNCTIONS
// ============================================
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function getTimestamp() {
    return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

// ============================================
// LOGGING SYSTEM
// ============================================
function addLog(message, type = 'info') {
    const logContainer = document.getElementById('activity-log');
    const timestamp = getTimestamp();

    const logEntry = document.createElement('div');
    const safeType = ['info', 'warning', 'danger'].includes(type) ? type : 'info';
    logEntry.className = `log-entry ${safeType}`;
    const time = document.createElement('span');
    time.className = 'log-time';
    time.textContent = timestamp;
    logEntry.append(time, document.createTextNode(String(message)));

    logContainer.insertBefore(logEntry, logContainer.firstChild);

    // Store in state
    state.proctoring.logs.push({ time: timestamp, message, type });

    // Keep only last 50 logs in DOM
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

// ============================================
// WARNING SYSTEM
// ============================================
function showWarning(title, message) {
    state.proctoring.warnings++;
    document.getElementById('warning-count').textContent = state.proctoring.warnings;

    addLog(`⚠️ WARNING: ${message}`, 'danger');

    // Show modal
    document.getElementById('warning-title').textContent = title;
    document.getElementById('warning-message').textContent = message;
    document.getElementById('warning-modal').classList.add('active');

    // At the threshold, preserve the student's exam and flag it for human review.
    if (!state.proctoring.needsManualReview &&
        ExamLifecycle.shouldRecommendManualReview(state.proctoring.warnings, state.proctoring.maxWarnings)) {
        state.proctoring.needsManualReview = true;
        addLog('Warning threshold reached; manual review recommended.', 'warning');
        document.getElementById('warning-title').textContent = 'Review Recommended';
        document.getElementById('warning-message').textContent =
            'The warning threshold was reached. Your exam will continue; this is not a finding of misconduct.';
    }
}

document.getElementById('dismiss-warning')?.addEventListener('click', () => {
    document.getElementById('warning-modal').classList.remove('active');
});

// ============================================
// CAMERA & MEDIAPIPE SETUP
// ============================================
async function initCamera() {
    try {
        state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        document.getElementById('preview-video').srcObject = state.cameraStream;
        document.getElementById('check-camera').classList.add('passed');
        document.getElementById('check-camera').classList.remove('failed');
        document.getElementById('check-camera').querySelector('.check-icon').textContent = '✅';
        updateStartAvailability();
        return true;
    } catch (err) {
        console.error('Camera error:', err);
        document.getElementById('check-camera').classList.add('failed');
        document.getElementById('check-camera').querySelector('.check-icon').textContent = '❌';
        updateStartAvailability();
        return false;
    }
}

async function initMicrophone() {
    try {
        state.microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        try {
            initAudioMonitor(state.microphoneStream);
        } catch (err) {
            console.error('Audio monitoring could not be initialized:', err);
        }
        document.getElementById('check-mic').classList.add('passed');
        document.getElementById('check-mic').querySelector('.check-icon').textContent = '✅';
        updateStartAvailability();
        return true;
    } catch (err) {
        console.error('Microphone error:', err);
        document.getElementById('check-mic').classList.add('failed');
        document.getElementById('check-mic').querySelector('.check-icon').textContent = '❌';
        updateStartAvailability();
        return false;
    }
}

async function initFaceMesh() {
    try {
        const faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceResults);
        state.face.mesh = faceMesh;
        return true;
    } catch (err) {
        console.error('FaceMesh error:', err);
        document.getElementById('check-face').classList.add('failed');
        document.getElementById('check-face').querySelector('.check-icon').textContent = '❌';
        updateStartAvailability();
        return false;
    }
}

function onFaceResults(results) {
    const canvas = state.isSetupComplete ?
        document.getElementById('exam-canvas') :
        document.getElementById('preview-canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        if (!state.face.faceDetected) {
            state.face.faceDetected = true;
            document.getElementById('check-face').classList.add('passed');
            document.getElementById('check-face').classList.remove('failed');
            document.getElementById('check-face').querySelector('.check-icon').textContent = '✅';
            document.getElementById('face-status').textContent = '✅ Face detected';
            updateStartAvailability();
        }
        const landmarks = results.multiFaceLandmarks[0];

        // Draw face mesh outline
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
        ctx.lineWidth = 1;

        // Get eye landmarks for gaze detection
        const leftEye = landmarks[468]; // Left eye center
        const rightEye = landmarks[473]; // Right eye center
        const nose = landmarks[1]; // Nose tip

        // Calculate gaze direction
        const eyeCenter = {
            x: (leftEye.x + rightEye.x) / 2,
            y: (leftEye.y + rightEye.y) / 2
        };

        // Check if looking away (eyes/nose position relative to center)
        const isLookingLeft = nose.x < 0.35;
        const isLookingRight = nose.x > 0.65;
        const isLookingUp = nose.y < 0.35;
        const isLookingDown = nose.y > 0.65;

        const isLookingAway = isLookingLeft || isLookingRight || isLookingUp || isLookingDown;

        // Update gaze indicator (smaller corner version)
        if (state.isSetupComplete) {
            const gazeDot = document.querySelector('.gaze-dot');

            // Move dot within small container based on gaze
            const offsetX = (nose.x - 0.5) * 30; // Smaller movement range
            const offsetY = (nose.y - 0.5) * 30;

            gazeDot.style.transform = `translate(${offsetX}px, ${offsetY}px)`;

            if (isLookingAway) {
                gazeDot.style.background = '#ef4444';
                gazeDot.style.boxShadow = '0 0 8px #ef4444';

                if (!state.face.isLookingAway) {
                    state.face.isLookingAway = true;
                    state.face.lookAwayStart = Date.now();
                }

                // Warn after 3 seconds of looking away
                if (Date.now() - state.face.lookAwayStart > 3000) {
                    state.proctoring.gazeAway++;
                    document.getElementById('stat-gaze').textContent = state.proctoring.gazeAway;
                    showWarning('Gaze Warning', 'Please keep your eyes on the screen during the exam.');
                    state.face.lookAwayStart = Date.now();
                }
            } else {
                gazeDot.style.background = '#6366f1';
                gazeDot.style.boxShadow = '0 0 6px #6366f1';
                state.face.isLookingAway = false;
            }
        }

        // Draw face bounding box
        let minX = 1, maxX = 0, minY = 1, maxY = 0;
        landmarks.forEach(landmark => {
            minX = Math.min(minX, landmark.x);
            maxX = Math.max(maxX, landmark.x);
            minY = Math.min(minY, landmark.y);
            maxY = Math.max(maxY, landmark.y);
        });

        ctx.strokeStyle = isLookingAway ? 'rgba(239, 68, 68, 0)' : 'rgba(16, 185, 129, 0)';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            minX * canvas.width,
            minY * canvas.height,
            (maxX - minX) * canvas.width,
            (maxY - minY) * canvas.height
        );

        // Face status
        if (!state.isSetupComplete) {
            document.getElementById('face-status').textContent = '✅ Face detected';
            document.getElementById('face-status').style.color = '#10b981';
        }
    } else {
        state.face.faceDetected = false;
        if (!state.isSetupComplete) {
            document.getElementById('check-face').classList.remove('passed');
            document.getElementById('check-face').querySelector('.check-icon').textContent = '⏳';
            document.getElementById('face-status').textContent = '⚠️ No face detected';
            document.getElementById('face-status').style.color = '#f59e0b';
            updateStartAvailability();
        } else if (!state.face.noFaceLogged) {
            state.face.noFaceLogged = true;
            addLog('Face not detected in frame', 'warning');
        }
    }

    if (results.multiFaceLandmarks?.length) state.face.noFaceLogged = false;
}

async function startFaceDetection(video) {
    if (!state.face.mesh || !video || !state.cameraStream) return false;
    if (state.mediaCamera) return true;
    const camera = new Camera(video, {
        onFrame: async () => {
            if (state.face.mesh) {
                await state.face.mesh.send({ image: video });
            }
        },
        width: 640,
        height: 480
    });
    state.mediaCamera = camera;
    state.face.video = video;
    try {
        await camera.start();
        return true;
    } catch (err) {
        console.error('Camera frame loop error:', err);
        try { camera.stop(); } catch (_) { /* camera loop was not started */ }
        state.mediaCamera = null;
        document.getElementById('check-face').classList.remove('passed');
        document.getElementById('check-face').classList.add('failed');
        document.getElementById('check-face').querySelector('.check-icon').textContent = '❌';
        updateStartAvailability();
        return false;
    }
}

// ============================================
// AUDIO MONITORING
// ============================================
function initAudioMonitor(stream) {
    if (!stream || state.audio.context) return;
    state.audio.context = new (window.AudioContext || window.webkitAudioContext)();
    state.audio.analyser = state.audio.context.createAnalyser();

    const source = state.audio.context.createMediaStreamSource(stream);
    source.connect(state.audio.analyser);

    state.audio.analyser.fftSize = 256;
    const bufferLength = state.audio.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    state.audio.running = true;

    function updateAudioVisualizer() {
        if (!state.audio.running) return;
        if (!state.isSetupComplete) {
            state.animationFrame = requestAnimationFrame(updateAudioVisualizer);
            return;
        }

        state.audio.analyser.getByteFrequencyData(dataArray);

        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
        }
        const average = sum / bufferLength;

        // Update visualizer bars
        const bars = document.querySelectorAll('.audio-bar');
        bars.forEach((bar, i) => {
            const value = dataArray[i * Math.floor(bufferLength / 5)] / 255;
            bar.style.height = `${5 + value * 35}px`;
        });

        // Volume can show audio activity, but cannot reliably identify speech.
        const audioStatus = document.getElementById('audio-status');
        if (average > 30) {
            if (!state.audio.isActive) {
                state.audio.isActive = true;
                audioStatus.textContent = 'Audio Activity';
                audioStatus.classList.add('active');
            }
        } else {
            if (state.audio.isActive) {
                state.audio.isActive = false;
                audioStatus.textContent = 'Silent';
                audioStatus.classList.remove('active');
            }
        }

        state.animationFrame = requestAnimationFrame(updateAudioVisualizer);
    }

    updateAudioVisualizer();
}

// ============================================
// FULLSCREEN HANDLING
// ============================================
let fullscreenWarningShown = false;

function fullscreenSupported() {
    const root = document.documentElement;
    const apiAvailable = !!(root.requestFullscreen || root.webkitRequestFullscreen || root.msRequestFullscreen);
    if (!apiAvailable) return false;
    const reportedCapabilities = [document.fullscreenEnabled, document.webkitFullscreenEnabled, document.msFullscreenEnabled]
        .filter(value => typeof value === 'boolean');
    if (reportedCapabilities.length) return reportedCapabilities.some(Boolean);
    return true;
}

function fullscreenActive() {
    return !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
}

function updateStartAvailability() {
    const name = document.getElementById('student-name')?.value.trim();
    const id = document.getElementById('student-id')?.value.trim();
    const ready = name && id && state.cameraStream && state.microphoneStream && state.face.faceDetected && fullscreenSupported();
    const button = document.getElementById('start-exam-btn');
    if (button) button.disabled = !ready;
}

async function requestFullscreen() {
    try {
        if (document.documentElement.requestFullscreen) {
            await document.documentElement.requestFullscreen();
        } else if (document.documentElement.webkitRequestFullscreen) {
            await document.documentElement.webkitRequestFullscreen();
        } else if (document.documentElement.msRequestFullscreen) {
            await document.documentElement.msRequestFullscreen();
        } else {
            return false;
        }
        if (!fullscreenActive()) return false;
        fullscreenWarningShown = false;
        return true;
    } catch (err) {
        console.error('Fullscreen error:', err);
        // Fullscreen might be blocked - continue anyway but log it
        addLog('Fullscreen request blocked by browser', 'warning');
        return false;
    }
}

document.addEventListener('fullscreenchange', () => {
    if (lifecycle.status === 'active' && !fullscreenActive() && !fullscreenWarningShown) {
        fullscreenWarningShown = true;
        state.proctoring.focusLosses++;
        document.getElementById('stat-focus').textContent = state.proctoring.focusLosses;
        addLog('🖥️ Exited fullscreen mode', 'danger');
        showWarning('Fullscreen Exit!', 'You exited fullscreen mode. Please stay in fullscreen during the exam.');

        // Try to re-enter fullscreen after a delay
        setTimeout(() => {
            if (lifecycle.status === 'active') requestFullscreen();
        }, 1000);
    } else if (fullscreenActive()) fullscreenWarningShown = false;
});
document.addEventListener('webkitfullscreenchange', () => {
    if (lifecycle.status === 'active' && !fullscreenActive() && !fullscreenWarningShown) {
        fullscreenWarningShown = true;
        state.proctoring.focusLosses++;
        document.getElementById('stat-focus').textContent = state.proctoring.focusLosses;
        showWarning('Fullscreen Exit!', 'You exited fullscreen mode. Please stay in fullscreen during the exam.');
    }
});
document.addEventListener('MSFullscreenChange', () => {
    if (lifecycle.status === 'active' && !fullscreenActive() && !fullscreenWarningShown) {
        fullscreenWarningShown = true;
        state.proctoring.focusLosses++;
        document.getElementById('stat-focus').textContent = state.proctoring.focusLosses;
        showWarning('Fullscreen Exit!', 'You exited fullscreen mode. Please stay in fullscreen during the exam.');
    } else if (fullscreenActive()) fullscreenWarningShown = false;
});

// Keyboard shortcut to re-enter fullscreen
document.addEventListener('keydown', (e) => {
    if (lifecycle.status === 'active' && e.key === 'F11') {
        e.preventDefault();
        requestFullscreen();
    }
});

// ============================================
// TAB SWITCHING & FOCUS DETECTION (ENHANCED)
// ============================================
let tabSwitchCount = 0;
let lastTabSwitchTime = null;
let isPageVisible = true;

// Create visual overlay for tab switch warning
function showTabSwitchOverlay() {
    let overlay = document.getElementById('tab-switch-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'tab-switch-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(239, 68, 68, 0.9);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            color: white;
            font-size: 24px;
            text-align: center;
            animation: fadeIn 0.3s ease;
        `;
        const icon = document.createElement('div');
        icon.style.cssText = 'font-size:80px;margin-bottom:20px';
        icon.textContent = '⚠️';
        const heading = document.createElement('h1');
        heading.style.marginBottom = '10px';
        heading.textContent = 'TAB SWITCH DETECTED!';
        const notice = document.createElement('p');
        notice.style.cssText = 'font-size:18px;opacity:0.9';
        notice.textContent = 'This violation has been recorded.';
        const dismiss = document.createElement('p');
        dismiss.style.cssText = 'font-size:16px;margin-top:20px;opacity:0.7';
        dismiss.textContent = 'Click anywhere to dismiss';
        overlay.replaceChildren(icon, heading, notice, dismiss);
        overlay.onclick = () => {
            overlay.remove();
        };
        document.body.appendChild(overlay);
    }

    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (overlay && overlay.parentNode) {
            overlay.remove();
        }
    }, 3000);
}

// Primary detection: Visibility API
document.addEventListener('visibilitychange', () => {
    if (!state.isSetupComplete) return;

    const wasVisible = isPageVisible;
    isPageVisible = !document.hidden;

    console.log('[PROCTOR] Visibility changed:', document.hidden ? 'HIDDEN' : 'VISIBLE');

    if (document.hidden) {
        // Page became hidden - user switched away
        tabSwitchCount++;
        state.proctoring.tabSwitches = tabSwitchCount;
        lastTabSwitchTime = new Date().toLocaleTimeString();
        console.log('[PROCTOR] TAB SWITCH #' + tabSwitchCount + ' at ' + lastTabSwitchTime);
    } else if (wasVisible === false) {
        // Page became visible again - show warning NOW
        document.getElementById('stat-tabs').textContent = state.proctoring.tabSwitches;
        addLog(`🚨 TAB SWITCH DETECTED (Switch #${tabSwitchCount} at ${lastTabSwitchTime})`, 'danger');

        // Show prominent overlay
        showTabSwitchOverlay();

        // Also show the modal warning
        showWarning('Tab Switch Detected!', `You switched away at ${lastTabSwitchTime}. This is violation #${tabSwitchCount}.`);
    }
});

// Secondary detection: Window blur/focus
let blurTime = null;
window.addEventListener('blur', () => {
    if (!state.isSetupComplete) return;
    blurTime = Date.now();
    console.log('[PROCTOR] Window blur at', new Date().toLocaleTimeString());
});

window.addEventListener('focus', () => {
    if (!state.isSetupComplete) return;

    // If we were blurred for more than 500ms, count as focus loss
    if (blurTime && (Date.now() - blurTime > 500)) {
        state.proctoring.focusLosses++;
        document.getElementById('stat-focus').textContent = state.proctoring.focusLosses;
        addLog('⚠️ Window focus lost and restored', 'warning');
    }
    blurTime = null;
});

// ============================================
// BEHAVIORAL LEARNING (Baseline + Anomaly Detection)
// ============================================
const behaviorBaseline = {
    typingIntervals: [],
    avgTypingSpeed: 0,
    typingVariance: 0,
    baselineEstablished: false,
    anomalyScore: 0
};

function updateBehaviorBaseline(interval) {
    behaviorBaseline.typingIntervals.push(interval);

    // Establish baseline after 20 keystrokes
    if (behaviorBaseline.typingIntervals.length >= 20 && !behaviorBaseline.baselineEstablished) {
        const intervals = behaviorBaseline.typingIntervals;
        behaviorBaseline.avgTypingSpeed = intervals.reduce((a, b) => a + b, 0) / intervals.length;

        // Calculate variance
        const squaredDiffs = intervals.map(i => Math.pow(i - behaviorBaseline.avgTypingSpeed, 2));
        behaviorBaseline.typingVariance = Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / intervals.length);

        behaviorBaseline.baselineEstablished = true;
        addLog(`📊 Behavior baseline established (avg: ${Math.round(behaviorBaseline.avgTypingSpeed)}ms, σ: ${Math.round(behaviorBaseline.typingVariance)}ms)`, 'info');
    }

    // Detect anomalies after baseline is established
    if (behaviorBaseline.baselineEstablished) {
        const deviation = Math.abs(interval - behaviorBaseline.avgTypingSpeed);
        const threshold = behaviorBaseline.typingVariance * 3; // 3 sigma rule

        if (deviation > threshold && interval < 30) {
            // Anomalously fast typing (potential automation or paste)
            behaviorBaseline.anomalyScore += 0.5;
            if (behaviorBaseline.anomalyScore > 5) {
                addLog('🤖 Anomalous typing pattern detected (too fast)', 'warning');
                behaviorBaseline.anomalyScore = 0;
            }
        }
    }
}

// Export behavior baseline for results
function getBehaviorScore() {
    if (!behaviorBaseline.baselineEstablished) return 100;

    const anomalyPenalty = Math.min(behaviorBaseline.anomalyScore * 5, 30);
    return Math.max(0, 100 - anomalyPenalty);
}

// ============================================
// CLIPBOARD MONITORING
// ============================================
document.addEventListener('paste', (e) => {
    if (state.isSetupComplete) {
        state.proctoring.pasteEvents++;
        document.getElementById('stat-paste').textContent = state.proctoring.pasteEvents;
        // Never read or log clipboard contents; record only the event count.
        addLog('Paste event detected', 'danger');

        if (state.proctoring.pasteEvents >= 2) {
            showWarning('Paste Detected', 'Multiple paste operations detected. This is flagged as suspicious behavior.');
        }
    }
});

// Prevent copy
document.addEventListener('copy', (e) => {
    if (state.isSetupComplete) {
        addLog('Copy attempt blocked', 'warning');
        e.preventDefault();
    }
});

// ============================================
// KEYSTROKE ANALYSIS (with behavioral learning)
// ============================================
function initKeystrokeAnalysis() {
    const answerInput = document.getElementById('answer-input');

    answerInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey || (e.key.length !== 1 && !['Backspace', 'Delete'].includes(e.key))) return;
        const now = Date.now();
        const questionIndex = state.exam.currentQuestion;
        if (state.exam.firstKeyDelaysMs[questionIndex] === null) {
            state.exam.firstKeyDelaysMs[questionIndex] = Math.max(0, now - state.exam.questionStartedAt);
        }
        state.typing.keystrokeCount++;
        if (e.key === 'Backspace') state.typing.backspaceCount++;
        if (e.key === 'Backspace' || e.key === 'Delete') {
            state.typing.editCount++;
        }
        if (e.key.length !== 1) return;

        if (state.typing.lastKeyTime > 0) {
            const interval = now - state.typing.lastKeyTime;
            if (interval > 0 && interval < 10000) {
                state.typing.keyIntervals.push(interval);
                state.typing.examKeyIntervals.push(interval);
                updateBehaviorBaseline(interval);

                if (interval < state.typing.burstThreshold) {
                    state.typing.burstCount++;
                    if (state.typing.burstCount >= 10) {
                        state.proctoring.typingBursts++;
                        document.getElementById('stat-bursts').textContent = state.proctoring.typingBursts;
                        addLog('⚡ Burst typing detected (suspiciously fast)', 'warning');
                        state.typing.burstCount = 0;
                    }
                } else {
                    state.typing.burstCount = Math.max(0, state.typing.burstCount - 1);
                }
            }
        }

        state.typing.lastKeyTime = now;
    });

    // Update typing stats display
    answerInput.addEventListener('input', () => {
        const text = answerInput.value;
        const now = Date.now();
        if (state.typing.lastInputTime > 0) {
            const idleSeconds = (now - state.typing.lastInputTime) / 1000;
            if (idleSeconds >= 2) state.typing.idleDurationsSeconds.push(idleSeconds);
        }
        state.typing.lastInputTime = now;
        document.getElementById('char-count').textContent = `${text.length} characters`;

        // Calculate WPM
        const wpm = ExamLifecycle.calculateWpm(state.typing.keyIntervals);
        document.getElementById('typing-speed').textContent = `${Math.min(wpm, 200)} WPM`;
    });
}

// ============================================
// QUESTION NAVIGATION
// ============================================
function loadQuestion(index) {
    const question = examQuestions[index];
    if (!question) return;
    state.typing.lastKeyTime = 0;
    state.typing.keyIntervals = [];
    state.exam.questionStartedAt = Date.now();
    const questionCard = document.getElementById('question-card');
    questionCard.replaceChildren();
    const topic = document.createElement('h3');
    topic.textContent = `Topic: ${question.topic}`;
    const text = document.createElement('p');
    text.textContent = question.text;
    questionCard.append(topic, text);

    // Load saved answer
    const answerInput = document.getElementById('answer-input');
    answerInput.value = state.exam.answers[index] || '';
    document.getElementById('char-count').textContent = `${answerInput.value.length} characters`;
    document.getElementById('typing-speed').textContent = '0 WPM';

    // Update navigation
    document.getElementById('current-q').textContent = index + 1;
    document.getElementById('prev-btn').disabled = index === 0;
    document.getElementById('next-btn').textContent = index === examQuestions.length - 1 ? 'Review answers →' : 'Next →';

    // Update indicators
    updateQuestionIndicators();

    addLog(`Viewing question ${index + 1}`, 'info');
}

function updateQuestionIndicators() {
    const container = document.getElementById('q-indicators');
    container.replaceChildren();

    examQuestions.forEach((q, i) => {
        const indicator = document.createElement('div');
        indicator.className = 'q-indicator';
        if (i === state.exam.currentQuestion) indicator.classList.add('current');
        if (state.exam.answers[i]) indicator.classList.add('answered');
        indicator.textContent = i + 1;
        indicator.onclick = () => {
            saveCurrentAnswer();
            questionNavigation.goTo(i);
            state.exam.currentQuestion = i;
            loadQuestion(i);
        };
        container.appendChild(indicator);
    });
}

function saveCurrentAnswer() {
    const answer = document.getElementById('answer-input').value;
    const index = state.exam.currentQuestion;
    state.exam.answers[index] = answer;
    if (state.exam.questionStartedAt !== null) {
        state.exam.questionDurationsMs[index] += Math.max(0, Date.now() - state.exam.questionStartedAt);
        state.exam.questionStartedAt = null;
    }
}

function openReview() {
    saveCurrentAnswer();
    const list = document.getElementById('review-question-list');
    list.replaceChildren();
    examQuestions.forEach((question, index) => {
        const row = document.createElement('div');
        row.className = 'review-question-row';
        const details = document.createElement('div');
        const heading = document.createElement('strong');
        heading.textContent = `Question ${index + 1}: ${question.topic}`;
        const prompt = document.createElement('p');
        prompt.textContent = question.text;
        const answerText = document.createElement('p');
        const answer = state.exam.answers[index] || '';
        answerText.textContent = `Your answer: ${answer.trim() || 'Not answered'}`;
        details.append(heading, prompt, answerText);
        const jump = document.createElement('button');
        jump.type = 'button';
        jump.className = 'nav-btn';
        jump.textContent = 'Review answer';
        jump.addEventListener('click', () => {
            state.exam.currentQuestion = index;
            questionNavigation.goTo(index);
            loadQuestion(index);
            showScreen('exam-screen');
        });
        row.append(details, jump);
        list.appendChild(row);
    });
    showScreen('review-screen');
}

// ============================================
// TIMER
// ============================================
// ============================================
// EXAM FLOW
// ============================================
async function startExam() {
    if (startPending || lifecycle.status !== 'setup') return;
    const name = document.getElementById('student-name').value.trim();
    const id = document.getElementById('student-id').value.trim();

    if (!name || !id) {
        alert('Please enter your name and student ID');
        return;
    }

    state.student.name = name;
    state.student.id = id;

    if (lifecycle.status !== 'setup') return;
    if (!state.cameraStream || !state.microphoneStream || !state.face.faceDetected) {
        alert('Camera, microphone, and face detection must be ready before starting.');
        return;
    }
    startPending = true;
    if (!fullscreenSupported() || !(fullscreenActive() || await requestFullscreen()) || !fullscreenActive()) {
        startPending = false;
        alert('Fullscreen mode is required to start the exam.');
        return;
    }

    try {
        const sessionResponse = await fetch('/api/exam-sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ student_name: name, student_id: id, question_count: examQuestions.length })
        });
        const session = await sessionResponse.json().catch(() => ({}));
        if (!sessionResponse.ok) throw new Error(session.message || 'Could not create a secure exam session');
        state.exam.sessionId = session.session_id;
        state.exam.sessionToken = session.session_token;

        const startResponse = await fetch(`/api/exam-sessions/${encodeURIComponent(session.session_id)}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${session.session_token}` },
            body: '{}'
        });
        if (!startResponse.ok) throw new Error('Could not activate the secure exam session');
    } catch (err) {
        startPending = false;
        state.exam.sessionId = null;
        state.exam.sessionToken = null;
        console.error('Secure exam session could not start:', err.message);
        alert('The exam could not be started securely. Check the server connection and try again.');
        return;
    }

    // Setup exam screen
    state.isSetupComplete = true;
    startPending = false;
    if (state.audio.context?.state === 'suspended') state.audio.context.resume();
    showScreen('exam-screen');

    // Setup video for exam
    document.getElementById('exam-video').srcObject = state.cameraStream;

    // Display student info
    document.getElementById('student-display').textContent = `${name} (${id})`;

    // Initialize
    document.getElementById('total-q').textContent = examQuestions.length;
    state.exam.answers = new Array(examQuestions.length).fill('');
    state.exam.submissionStatus = 'not-started';
    state.exam.verificationComplete = false;
    state.exam.verificationSaveStatus = 'not-started';
    state.exam.questionDurationsMs = new Array(examQuestions.length).fill(0);
    state.exam.firstKeyDelaysMs = new Array(examQuestions.length).fill(null);
    state.exam.questionStartedAt = null;
    state.typing.lastKeyTime = 0;
    state.typing.keyIntervals = [];
    state.typing.examKeyIntervals = [];
    state.typing.backspaceCount = 0;
    state.typing.keystrokeCount = 0;
    state.typing.editCount = 0;
    state.typing.idleDurationsSeconds = [];
    state.typing.lastInputTime = 0;
    state.behaviorPrediction = { status: 'not-requested', result: null };
    questionNavigation = ExamLifecycle.createQuestionNavigation(examQuestions.length);
    loadQuestion(0);
    initKeystrokeAnalysis();

    // Start timer
    state.exam.startTime = Date.now();
    lifecycle.start(state.exam.duration);

    addLog('Exam started', 'info');
}

function endExam(terminated = false) {
    if (!lifecycle.submit()) return false;
    saveCurrentAnswer();
    state.verification.responses = new Array(verificationQuestions.length).fill('');
    submitExamSession();
    requestBehaviorPrediction();

    if (terminated) {
        addLog('Exam terminated due to violations', 'danger');
    } else {
        addLog('Exam submitted', 'info');
    }

    // Go to verification or results
    startVerification();
    return true;
}

async function submitExamSession() {
    if (state.exam.submissionStatus === 'pending' || state.exam.submissionStatus === 'saved') return;
    if (!state.exam.sessionId || !state.exam.sessionToken) {
        state.exam.submissionStatus = 'failed';
        renderExamSubmissionStatus();
        return;
    }
    state.exam.submissionStatus = 'pending';
    renderExamSubmissionStatus();
    const payload = {
        answers: [...state.exam.answers],
        proctoring: {
            warnings: state.proctoring.warnings,
            tab_switches: state.proctoring.tabSwitches,
            focus_losses: state.proctoring.focusLosses,
            paste_events: state.proctoring.pasteEvents,
            gaze_away: state.proctoring.gazeAway,
            typing_bursts: state.proctoring.typingBursts,
            manual_review_recommended: state.proctoring.needsManualReview
        },
        verification_responses: [...state.verification.responses]
    };
    try {
        const response = await fetch(`/api/exam-sessions/${encodeURIComponent(state.exam.sessionId)}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${state.exam.sessionToken}` },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('submission_failed');
        state.exam.submissionStatus = 'saved';
        if (state.exam.verificationComplete) saveVerificationResponses();
    } catch (err) {
        console.error('Secure exam submission failed. Student data was not logged.', err.message);
        state.exam.submissionStatus = 'failed';
    }
    renderExamSubmissionStatus();
}

async function saveVerificationResponses() {
    if (state.exam.verificationSaveStatus === 'pending' || state.exam.verificationSaveStatus === 'saved') return;
    if (state.exam.submissionStatus !== 'saved' || !state.exam.sessionId || !state.exam.sessionToken) return;
    state.exam.verificationSaveStatus = 'pending';
    renderExamSubmissionStatus();
    try {
        const response = await fetch(`/api/exam-sessions/${encodeURIComponent(state.exam.sessionId)}/verification`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${state.exam.sessionToken}` },
            body: JSON.stringify({ verification_responses: [...state.verification.responses] })
        });
        if (!response.ok) throw new Error('verification_save_failed');
        state.exam.verificationSaveStatus = 'saved';
        state.exam.sessionToken = null;
    } catch (err) {
        console.error('Oral verification could not be saved.', err.message);
        state.exam.verificationSaveStatus = 'failed';
    }
    renderExamSubmissionStatus();
}

function renderExamSubmissionStatus() {
    const status = document.getElementById('exam-submission-status');
    const retry = document.getElementById('retry-exam-submit-btn');
    if (!status || !retry) return;
    if (state.exam.submissionStatus === 'pending') {
        status.textContent = 'Saving your exam securely…';
        retry.hidden = true;
    } else if (state.exam.submissionStatus === 'failed') {
        status.textContent = 'The server has not confirmed saving this exam. Retry when the connection is available.';
        retry.hidden = false;
    } else if (state.exam.submissionStatus === 'saved' && !state.exam.verificationComplete) {
        status.textContent = 'Exam answers saved securely. Oral verification is still in progress.';
        retry.hidden = true;
    } else if (state.exam.submissionStatus === 'saved' && state.exam.verificationSaveStatus === 'pending') {
        status.textContent = 'Saving oral verification securely…';
        retry.hidden = true;
    } else if (state.exam.submissionStatus === 'saved' && state.exam.verificationSaveStatus === 'failed') {
        status.textContent = 'Exam saved, but oral verification was not confirmed. Retry when the connection is available.';
        retry.hidden = false;
    } else if (state.exam.submissionStatus === 'saved' && state.exam.verificationSaveStatus === 'saved') {
        status.textContent = 'Your exam submission was saved.';
        retry.hidden = true;
    } else {
        status.textContent = 'Waiting for secure exam submission.';
        retry.hidden = true;
    }
}

function collectBehaviorFeatures() {
    return ExamLifecycle.calculateBehaviorFeatures({
        keyIntervalsMs: state.typing.examKeyIntervals,
        backspaceCount: state.typing.backspaceCount,
        keystrokeCount: state.typing.keystrokeCount,
        pasteCount: state.proctoring.pasteEvents,
        tabSwitches: state.proctoring.tabSwitches,
        focusLosses: state.proctoring.focusLosses,
        idleDurationsSeconds: state.typing.idleDurationsSeconds,
        editCount: state.typing.editCount,
        firstKeyDelaysSeconds: state.exam.firstKeyDelaysMs.filter(value => value !== null).map(value => value / 1000),
        questionCount: state.exam.answers.length || 1,
        answerDurationSeconds: state.exam.questionDurationsMs.reduce((sum, value) => sum + value, 0) / 1000
    });
}

async function requestBehaviorPrediction() {
    if (state.behaviorPrediction.status !== 'not-requested') return;
    state.behaviorPrediction = { status: 'pending', result: null };
    renderBehaviorPrediction();
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: collectBehaviorFeatures() })
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || 'prediction_unavailable');
        state.behaviorPrediction = { status: 'ready', result: body };
    } catch (error) {
        console.warn('Behavior model prediction unavailable:', error.message);
        state.behaviorPrediction = { status: 'unavailable', result: null };
    }
    renderBehaviorPrediction();
}

function renderBehaviorPrediction() {
    const container = document.getElementById('ml-prediction-summary');
    if (!container) return;
    container.replaceChildren();
    const status = state.behaviorPrediction.status;
    const message = document.createElement('p');
    if (status === 'pending') {
        message.textContent = 'Evaluating the aggregate behavior summary…';
    } else if (status !== 'ready') {
        message.textContent = 'Model result unavailable. The existing proctoring summary remains available.';
    } else {
        const result = state.behaviorPrediction.result;
        message.textContent = result.alert_category === 'review'
            ? 'The model flagged this summary for human review. This is not proof of cheating.'
            : 'The model did not flag this summary.';
        const score = document.createElement('p');
        score.textContent = `Isolation Forest decision score: ${result.decision_score} (lower values are more anomalous).`;
        container.appendChild(score);
    }
    container.prepend(message);
}

// ============================================
// ORAL VERIFICATION
// ============================================
let recognition = null;

function startVerification() {
    showScreen('verification-screen');
    document.getElementById('verification-video').srcObject = state.cameraStream;
    state.verification.currentQuestion = 0;
    state.verification.responses = new Array(verificationQuestions.length).fill('');

    // Setup speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) finalTranscript += result[0].transcript.trim() + ' ';
                else interimTranscript += result[0].transcript;
            }
            state.verification.responses[state.verification.currentQuestion] = finalTranscript.trim();
            document.getElementById('transcript-output').textContent = `${finalTranscript}${interimTranscript}`.trim();
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            if (event.error !== 'aborted') {
                document.getElementById('transcript-output').textContent = event.error === 'not-allowed'
                    ? 'Microphone permission was denied. You can try again or continue.'
                    : `Speech recognition error: ${event.error}. You can try again or continue.`;
            }
            if (recognitionActive) {
                try { recognition.stop(); } catch (_) { /* onend will complete the attempt */ }
            }
        };
        recognition.onend = finishRecognition;
    } else {
        alert('Speech recognition is not supported. Skipping verification.');
        showResults();
        return;
    }

    loadVerificationQuestion(0);
}

function loadVerificationQuestion(index) {
    if (index < 0 || index >= verificationQuestions.length) return;
    const container = document.getElementById('verification-question');
    container.querySelector('.v-question-number').textContent =
        `Question ${index + 1} of ${verificationQuestions.length}`;
    document.getElementById('v-question-text').textContent = verificationQuestions[index];

    // Reset UI
    finalTranscript = state.verification.responses[index] || '';
    document.getElementById('transcript-output').textContent = finalTranscript || 'Click "Start Speaking" to begin...';
    document.getElementById('recording-status').classList.remove('active');
    document.getElementById('start-speaking').style.display = 'flex';
    document.getElementById('next-v-question').style.display = 'none';
    document.getElementById('complete-verification').style.display = 'none';
    if (!recognition && index === verificationQuestions.length - 1) document.getElementById('complete-verification').style.display = 'block';
}

function finishRecognition() {
    if (!recognitionActive) return;
    recognitionActive = false;
    if (verificationTimeout !== null) clearTimeout(verificationTimeout);
    verificationTimeout = null;
    const transcript = finalTranscript.trim();
    state.verification.responses[state.verification.currentQuestion] = transcript;
    document.getElementById('recording-status').classList.remove('active');
    if (!transcript) {
        document.getElementById('start-speaking').style.display = 'flex';
        document.getElementById('transcript-output').textContent = 'No speech captured. You can try again or continue.';
        if (state.verification.currentQuestion < verificationQuestions.length - 1) {
            document.getElementById('next-v-question').style.display = 'block';
        } else {
            document.getElementById('complete-verification').style.display = 'block';
        }
        return;
    }
    document.getElementById('transcript-output').textContent = transcript;
    document.getElementById('start-speaking').style.display = 'none';
    if (state.verification.currentQuestion < verificationQuestions.length - 1) {
        document.getElementById('next-v-question').style.display = 'block';
    } else {
        document.getElementById('complete-verification').style.display = 'block';
    }
}

function stopRecognition() {
    if (verificationTimeout !== null) clearTimeout(verificationTimeout);
    verificationTimeout = null;
    if (recognition && recognitionActive) {
        try { recognition.stop(); } catch (err) { console.warn('Speech recognition stop failed:', err); }
    }
}

document.getElementById('start-speaking')?.addEventListener('click', () => {
    if (recognition && !recognitionActive) {
        finalTranscript = '';
        state.verification.responses[state.verification.currentQuestion] = '';
        try {
            recognition.start();
            recognitionActive = true;
        } catch (err) {
            console.error('Speech recognition start failed:', err);
            document.getElementById('transcript-output').textContent = 'Could not start speech recognition. Please try again.';
            return;
        }
        document.getElementById('recording-status').classList.add('active');
        document.getElementById('start-speaking').style.display = 'none';
        document.getElementById('transcript-output').textContent = 'Listening...';
        verificationTimeout = setTimeout(stopRecognition, 30000);
    }
});

document.getElementById('next-v-question')?.addEventListener('click', () => {
    state.verification.currentQuestion++;
    loadVerificationQuestion(state.verification.currentQuestion);
});

document.getElementById('complete-verification')?.addEventListener('click', () => {
    stopRecognition();
    showResults();
});

function cleanupSessionResources() {
    state.isSetupComplete = false;
    stopRecognition();
    if (recognition) {
        recognition.onend = null;
        recognition.onerror = null;
        recognition.onresult = null;
        recognition = null;
    }
    recognitionActive = false;
    lifecycle.complete();
    lifecycle.dispose();
    state.audio.running = false;
    if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
    if (state.mediaCamera) {
        try { state.mediaCamera.stop(); } catch (err) { console.warn('Camera loop stop failed:', err); }
        state.mediaCamera = null;
    }
    [state.cameraStream, state.microphoneStream].filter(Boolean).forEach(stream => {
        stream.getTracks().forEach(track => track.stop());
    });
    state.cameraStream = null;
    state.microphoneStream = null;
    if (state.audio.context && state.audio.context.state !== 'closed') {
        state.audio.context.close().catch(err => console.warn('Audio context close failed:', err));
    }
    state.audio.context = null;
    state.audio.analyser = null;
    if (state.face.mesh && state.face.mesh.close) {
        try {
            const closing = state.face.mesh.close();
            closing?.catch(err => console.warn('Face Mesh close failed:', err));
        } catch (err) {
            console.warn('Face Mesh close failed:', err);
        }
    }
    state.face.mesh = null;
    document.querySelectorAll('video').forEach(video => { video.srcObject = null; });
}

// ============================================
// RESULTS
// ============================================
function showResults() {
    if (lifecycle.status === 'completed') return;
    state.exam.verificationComplete = true;
    if (state.exam.submissionStatus === 'saved') saveVerificationResponses();
    else submitExamSession();
    cleanupSessionResources();
    showScreen('results-screen');

    // Exit fullscreen
    if (fullscreenActive()) {
        (document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen)?.call(document);
    }

    // Calculate integrity score
    const totalViolations = state.proctoring.warnings +
        state.proctoring.tabSwitches +
        state.proctoring.pasteEvents +
        Math.floor(state.proctoring.gazeAway / 2);

    const score = Math.max(0, 100 - (totalViolations * 15));

    // Update summary
    const summaryGrid = document.getElementById('summary-grid');
    const stats = [
        ['Tab Switches', state.proctoring.tabSwitches],
        ['Focus Lost', state.proctoring.focusLosses],
        ['Paste Events', state.proctoring.pasteEvents],
        ['Gaze Warnings', state.proctoring.gazeAway],
        ['Typing Bursts', state.proctoring.typingBursts],
        ['Total Warnings', state.proctoring.warnings],
        ['Manual Review', state.proctoring.needsManualReview ? 'Recommended' : 'Not flagged']
    ];
    summaryGrid.replaceChildren(...stats.map(([label, value]) => {
        const item = document.createElement('div');
        item.className = 'summary-item';
        const number = document.createElement('div');
        number.className = 'value';
        number.textContent = String(value);
        const caption = document.createElement('div');
        caption.className = 'label';
        caption.textContent = label;
        item.append(number, caption);
        return item;
    }));

    // Update score
    const scoreCircle = document.getElementById('score-circle');
    document.getElementById('integrity-score').textContent = `${score}%`;

    if (score >= 80) {
        scoreCircle.className = 'score-circle';
        document.getElementById('score-message').textContent = 'Excellent integrity! No significant issues detected.';
    } else if (score >= 50) {
        scoreCircle.className = 'score-circle warning';
        document.getElementById('score-message').textContent = 'Some suspicious activity detected. Manual review recommended.';
    } else {
        scoreCircle.className = 'score-circle danger';
        document.getElementById('score-message').textContent = 'Multiple violations detected. Exam flagged for review.';
    }

    // Warning log
    const warningLogDiv = document.getElementById('warning-log-summary');
    warningLogDiv.replaceChildren();
    const warningLogs = state.proctoring.logs.filter(log => log.type !== 'info').slice(0, 10);
    if (warningLogs.length) {
        warningLogs.forEach(log => {
            const item = document.createElement('div');
            item.className = 'warning-log-item';
            const message = document.createElement('span');
            message.textContent = log.message;
            const time = document.createElement('span');
            time.textContent = log.time;
            item.append(message, time);
            warningLogDiv.appendChild(item);
        });
    } else {
        const empty = document.createElement('p');
        empty.style.cssText = 'text-align:center;color:#64748b';
        empty.textContent = 'No warnings recorded';
        warningLogDiv.appendChild(empty);
    }

    const oralResponses = document.getElementById('oral-responses-summary');
    oralResponses.replaceChildren();
    state.verification.responses.forEach((response, index) => {
        const item = document.createElement('div');
        item.className = 'warning-log-item';
        const text = document.createElement('span');
        text.textContent = `Question ${index + 1}: ${response || 'No response captured'}`;
        item.appendChild(text);
        oralResponses.appendChild(item);
    });
}

document.getElementById('restart-btn')?.addEventListener('click', () => {
    location.reload();
});

document.getElementById('retry-exam-submit-btn')?.addEventListener('click', () => {
    if (state.exam.submissionStatus === 'saved') saveVerificationResponses();
    else submitExamSession();
});

// ============================================
// NAVIGATION HANDLERS
// ============================================
document.getElementById('prev-btn')?.addEventListener('click', () => {
    saveCurrentAnswer();
    if (state.exam.currentQuestion > 0) {
        questionNavigation.previous();
        state.exam.currentQuestion = questionNavigation.index;
        loadQuestion(state.exam.currentQuestion);
    }
});

document.getElementById('next-btn')?.addEventListener('click', () => {
    if (lifecycle.status !== 'active') return;
    saveCurrentAnswer();
    const next = questionNavigation.next();
    state.exam.currentQuestion = next.index;
    if (next.mode === 'review') openReview();
    else {
        loadQuestion(state.exam.currentQuestion);
    }
});

document.getElementById('review-back-btn')?.addEventListener('click', () => {
    questionNavigation.previous();
    state.exam.currentQuestion = questionNavigation.index;
    loadQuestion(state.exam.currentQuestion);
    showScreen('exam-screen');
});

document.getElementById('review-submit-btn')?.addEventListener('click', () => {
    if (confirm('Submit your exam now?')) endExam(false);
});

document.getElementById('submit-exam-btn')?.addEventListener('click', () => {
    if (lifecycle.status === 'active' && confirm('Are you sure you want to submit your exam?')) {
        endExam(false);
    }
});

// ============================================
// INITIALIZATION
// ============================================
async function init() {
    // Fetch questions from API
    await fetchQuestions();

    // Check inputs for enabling start button
    document.getElementById('student-name').addEventListener('input', updateStartAvailability);
    document.getElementById('student-id').addEventListener('input', updateStartAvailability);

    // Request camera and microphone independently so one denial does not mask the other.
    await Promise.all([initCamera(), initMicrophone(), initFaceMesh()]);

    // Start preview face detection
    const previewVideo = document.getElementById('preview-video');
    previewVideo.onloadedmetadata = () => {
        startFaceDetection(previewVideo);
    };
    if (previewVideo.readyState >= 1) startFaceDetection(previewVideo);

    if (fullscreenSupported()) {
        document.getElementById('check-fullscreen').classList.add('passed');
        document.getElementById('check-fullscreen').classList.remove('failed');
        document.getElementById('check-fullscreen').querySelector('.check-icon').textContent = '✅';
    } else {
        document.getElementById('check-fullscreen').classList.add('failed');
        document.getElementById('check-fullscreen').querySelector('.check-icon').textContent = '❌';
    }

    updateStartAvailability();
}

// Start button handler
document.getElementById('start-exam-btn')?.addEventListener('click', startExam);

// Initialize on load
window.addEventListener('load', init);

// Prevent right-click context menu during exam
document.addEventListener('contextmenu', (e) => {
    if (state.isSetupComplete) {
        e.preventDefault();
        addLog('Right-click blocked', 'info');
    }
});

// Prevent keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (state.isSetupComplete) {
        // Block common shortcuts
        if (e.ctrlKey && (e.key === 'c' || e.key === 'v' || e.key === 'a' || e.key === 'p')) {
            if (e.key !== 'v') { // Allow paste for detection
                e.preventDefault();
                addLog(`Keyboard shortcut Ctrl+${e.key.toUpperCase()} blocked`, 'info');
            }
        }

        // Block Alt+Tab indicator
        if (e.altKey && e.key === 'Tab') {
            addLog('Alt+Tab attempted', 'warning');
        }

        // Block F12, Ctrl+Shift+I
        if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && e.key === 'I')) {
            e.preventDefault();
            addLog('Developer tools blocked', 'warning');
        }
    }
});

// ============================================
// THEME & UI HANDLING
// ============================================
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.body.classList.add('dark-theme');
    }

    const toggleTheme = () => {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };

    // Attach listeners to all theme buttons
    document.querySelectorAll('.theme-toggle').forEach(btn => {
        btn.addEventListener('click', toggleTheme);
    });

    // Log Sidebar Toggle
    const toggleLogBtn = document.getElementById('toggle-log-btn');
    const logSidebar = document.getElementById('log-sidebar');
    
    toggleLogBtn?.addEventListener('click', () => {
        logSidebar.classList.toggle('collapsed');
        // Optional: save preference
    });
}

// Initialize theme immediately
initTheme();
