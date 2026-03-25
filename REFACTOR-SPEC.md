# agentic-sense Library Refactor Spec

## Goal
Refactor agentic-sense from a demo app into an independent, reusable library following the agentic family pattern (agentic-core, agentic-render).

## Agentic Family Conventions
- **Single-file core** — one JS file is the library (`agentic-sense.js`)
- **Zero runtime dependencies** — MediaPipe WASM/models are loaded at runtime by the consumer
- **ESM exports** — `export class AgenticSense { ... }`
- **Browser-first** — works with a single `<script type="module">` import
- **Separate demo** — `demo/` folder with example usage
- **Clean package.json** — name, version, description, main, browser, files, license MIT

## Architecture

### Library (`agentic-sense.js`) — THE deliverable
Single file that exports:

```js
export class AgenticSense {
  constructor(videoElement, options = {})
  async init(options)      // load MediaPipe models
  detect()                 // returns SenseFrame
  destroy()                // cleanup
  
  // Static
  static get VERSION()
}

// SenseFrame type (returned by detect()):
// {
//   face: { landmarks, expression, headPose, gaze, eyeState, attention },
//   hands: { left, right, gesture },
//   pose: { landmarks, torso },
//   objects: [{ name, confidence, bbox }],
//   meta: { timestamp, fps, activeModels }
// }
```

### What goes INTO agentic-sense.js:
- EMA utility class
- BlinkDetector
- ExpressionClassifier  
- HeadPoseEstimator
- GazeEstimator
- FocusScorer
- SynthesisEngine (renamed internally)
- SenseData extraction (merge sense-data.js INTO the library)
- IDX constants
- SenseEngine → renamed to AgenticSense

### What stays OUT (demo only):
- dashboard.js → `demo/dashboard.js`
- main.js → `demo/main.js`  
- index.html → `demo/index.html`
- style.css → `demo/style.css`
- Canvas overlay drawing code → `demo/overlay.js`

### Key design decisions:
1. **No canvas overlay in library** — library returns data, consumer draws
2. **MediaPipe path configurable** — `init({ wasmPath: './mediapipe/', modelPath: './mediapipe/' })`
3. **Selective model loading** — `init({ face: true, hands: true, pose: false, objects: false })`
4. **No DOM manipulation** — library never touches DOM except reading video frames
5. **Overlay drawing extracted to demo** — the library's `detect()` returns raw landmarks, demo draws them

## File Structure After Refactor

```
agentic-sense/
├── agentic-sense.js       # THE library (single file)
├── package.json           # npm package metadata
├── README.md              # Usage docs
├── demo/
│   ├── index.html         # Demo app
│   ├── main.js            # Demo entry
│   ├── dashboard.js       # HUD display
│   ├── overlay.js         # Canvas overlay drawing
│   └── style.css          # Demo styles
├── mediapipe/             # Self-hosted WASM + models (gitignored, consumer provides)
└── .gitignore
```

## package.json

```json
{
  "name": "agentic-sense",
  "version": "0.1.0",
  "description": "Real-time human perception engine — face, hands, pose, objects. Zero dependencies, browser-first.",
  "type": "module",
  "main": "agentic-sense.js",
  "browser": "agentic-sense.js",
  "files": ["agentic-sense.js", "README.md"],
  "keywords": ["mediapipe", "face-detection", "pose-estimation", "hand-tracking", "perception", "computer-vision", "browser", "zero-dependency"],
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/momomo-agent/agentic-sense.git"
  }
}
```

## Constraints
- agentic-sense.js MUST be under 800 lines (current engine.js is 862 + sense-data.js 415 = ~1200 total, need to trim)
- Strip ALL canvas drawing code from the library
- Strip ALL DOM references except video element reading
- Keep the overlay/drawing code in demo/overlay.js
- The demo MUST still work after refactor (same visual result)
- Do NOT modify the mediapipe/ directory
- Do NOT delete any git history

## Consumer Usage Example (for README)

```html
<video id="cam" autoplay></video>
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
    hands: true 
  })
  
  function loop() {
    const frame = sense.detect()
    if (frame?.face) {
      console.log('Expression:', frame.face.expression)
      console.log('Attention:', frame.face.attention)
    }
    requestAnimationFrame(loop)
  }
  loop()
</script>
```
