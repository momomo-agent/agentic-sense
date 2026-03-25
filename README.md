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

## Features

- **478 face landmarks** with iris tracking
- **52 blendshapes** — every facial muscle as 0-1 weight
- **Interpreted data** — expression, focus, gaze, blink rate, distance
- **Multi-face** — up to 3 faces simultaneously
- **Hand tracking** — 21 landmarks per hand, gesture recognition
- **Body pose** — 33 skeletal landmarks
- **Object detection** — 80 COCO classes
- **Fully local** — zero network requests after model load
- **Single file** — ~480 lines, zero dependencies

## License

MIT
