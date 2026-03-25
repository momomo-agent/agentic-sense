# agentic-sense

Human perception for AI agents.

Face detection, attention tracking, expression, pose, hands, objects — all running locally in the browser via MediaPipe. Zero dependencies, single file.

Part of the [agentic.js](https://momomo-agent.github.io/agentic/) family.

## Quick Start

```html
<video id="cam" autoplay playsinline muted></video>
<script type="module">
  import { AgenticSense } from './agentic-sense.js'

  const video = document.getElementById('cam')
  const stream = await navigator.mediaDevices.getUserMedia({ video: true })
  video.srcObject = stream
  await video.play()

  const sense = new AgenticSense(video)
  await sense.init({
    wasmPath: './mediapipe/',
    face: true,
    hands: true,
  })

  function loop() {
    const frame = sense.detect()
    if (frame?.faceCount > 0) {
      const face = frame.faces[0]
      console.log('Facing:', face.head.facing)
      console.log('Expression:', face.interpretation?.expression)
      console.log('Focus:', face.interpretation?.focus)
      console.log('Blendshapes:', face.blendshapes)
    }
    requestAnimationFrame(loop)
  }
  loop()
</script>
```

### Audio (Speech + Wake Word)

```html
<script type="module">
  import { AgenticAudio } from './agentic-sense.js'

  const audio = new AgenticAudio({
    wakeWords: ['hey momo', 'momo'],
  })

  audio.onResult = (text, isFinal, judgment) => {
    console.log(text, judgment.isWake ? 'WAKE' : '')
  }

  // Connect to AgenticSense for multi-modal wake detection
  audio.onWake = (word, text, judgment) => {
    console.log('Wake:', word, judgment.reason)
  }

  await audio.start()

  // Feed visual context from sense (optional, enhances wake detection)
  function loop() {
    const frame = sense.detect()
    if (frame?.faces?.[0]) {
      audio.updateVisualContext(frame.faces[0].head.facing)
    }
    requestAnimationFrame(loop)
  }
</script>
```

## API

### `new AgenticSense(videoElement)`

Create a sense instance attached to a video element.

### `sense.init(options)`

Load MediaPipe models. Options:

| Option | Default | Description |
|--------|---------|-------------|
| `wasmPath` | `'./mediapipe/'` | Path to MediaPipe WASM + model files |
| `face` | `true` | Enable face landmarks (478 points + blendshapes) |
| `hands` | `false` | Enable hand tracking (gesture + landmarks) |
| `pose` | `false` | Enable body pose (33 landmarks) |
| `segment` | `false` | Enable person segmentation |
| `objects` | `false` | Enable object detection (80 COCO classes) |

### `sense.detect()` → `SenseFrame`

Detect one frame. Call in `requestAnimationFrame`.

Returns:

```js
{
  timestamp: 12345.67,
  faceCount: 1,
  faces: [{
    head: { yaw, pitch, roll, facing, faceWidth, faceHeight },
    eyes: { left, right, avgEAR, ipd, iris },
    mouth: { openness, width, ratio },
    blendshapes: { jawOpen, smileL, smileR, ... },  // 38 values
    interpretation: {  // only on primary face
      pose: { yaw, pitch, roll, facing },
      gaze: { region, looking, x, y },
      blinkRate: 14,
      focus: { score: 82, level: 'high' },
      expression: 'smiling',
      distance: 0.8,
    }
  }],
  handCount: 0,
  hands: [],
  customGestures: [],  // [{ hand, name, confidence, emoji }] — OK, Pinch, Rock, One, Peace
  actions: [],         // [{ hand, name, confidence, emoji }] — Wave, Tap, Push Down, Lift Up, Circle
  body: null,
  segmentation: null,
  objectCount: 0,
  objects: [],
}
```

### `sense.rawResults`

Access raw MediaPipe results for overlay drawing in your own canvas.

```js
const raw = sense.rawResults
// raw.face   — FaceLandmarker result (faceLandmarks, faceBlendshapes)
// raw.hand   — GestureRecognizer result (landmarks, gestures, handednesses)
// raw.handLm — HandLandmarker result (if GestureRecognizer unavailable)
// raw.pose   — PoseLandmarker result (landmarks)
// raw.seg    — ImageSegmenter result (categoryMask)
// raw.objects — ObjectDetector result (detections)
```

### `sense.destroy()`

Cleanup all MediaPipe instances.

### `new AgenticAudio(options)`

Create an audio sense instance. Options:

| Option | Default | Description |
|--------|---------|-------------|
| `wakeWords` | `['hello','hey momo','momo']` | Wake word list (case insensitive) |
| `lang` | `'zh-CN'` | Language for transcription |
| `serverUrl` | `null` | SenseVoice server URL (auto-detects `localhost:18906` if not set) |

### `audio.start()` → `Promise<boolean>`

Start microphone capture. Auto-detects a local SenseVoice server — if found, uses it (fast + accurate). Otherwise falls back to Whisper WASM in browser (slower, works anywhere).

### SenseVoice Server (optional, recommended)

For much faster and more accurate Chinese speech recognition, run the SenseVoice server locally:

```bash
cd server && ./setup.sh
```

This installs SenseVoice (~900MB model, first-run only) and starts a server on port 18906. Auto-detects Apple MPS / NVIDIA CUDA / CPU.

### `audio.stop()`

Stop capture, terminate worker, release microphone.

### `audio.updateVisualContext(facing)`

Feed camera data for multi-modal wake detection. `facing` is a boolean from `face.head.facing`.

### Callbacks

| Callback | Arguments | Description |
|----------|-----------|-------------|
| `onResult` | `(text, isFinal, judgment)` | Speech transcription result |
| `onWake` | `(wakeWord, fullText, judgment)` | Wake word detected (confidence >= 0.5) |
| `onVolumeChange` | `(volume)` | Microphone RMS level (0-1) |
| `onModelStatus` | `(status, message)` | Model loading status (`'loading'`, `'ready'`, `'error'`) |

## Features

- **478 face landmarks** with iris tracking
- **52 blendshapes** — every facial muscle as 0-1 weight
- **Interpreted data** — expression, focus, gaze, blink rate, distance
- **Multi-face** — up to 3 faces simultaneously
- **Hand tracking** — 21 landmarks per hand, gesture recognition
- **Custom gestures** — OK, Pinch, Rock, One, Peace (landmark-based fallback)
- **Action detection** — Wave, Tap, Push Down, Lift Up, Circle (temporal)
- **Speech recognition** — SenseVoice server (fast, auto-detected) or Whisper WASM fallback (works anywhere)
- **Wake word detection** — multi-modal fusion (audio + visual context)
- **Body pose** — 33 skeletal landmarks
- **Object detection** — 80 COCO classes
- **Fully local** — zero network requests after model load
- **Single file** — ~480 lines, zero dependencies
- **Audio module** — separate file with Web Worker for speech

## License

MIT
