import { AgenticSense } from '../agentic-sense.js'
import { drawOverlay } from './overlay.js'
import { Dashboard } from './dashboard.js'

const video = document.getElementById('camera')
const overlay = document.getElementById('overlay')
const statusDot = document.getElementById('status-dot')

let sense = null
let currentStream = null

async function enumerateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices()
  return devices.filter(d => d.kind === 'videoinput')
}

function populateSelect(selectEl, cameras, currentId) {
  selectEl.innerHTML = cameras.map((cam, i) =>
    `<option value="${cam.deviceId}" ${cam.deviceId === currentId ? 'selected' : ''}>${cam.label || `摄像头 ${i + 1}`}</option>`
  ).join('')
}

async function startCamera(deviceId) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop())
  }
  const constraints = {
    video: {
      width: 640, height: 480,
      ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' })
    }
  }
  const stream = await navigator.mediaDevices.getUserMedia(constraints)
  currentStream = stream
  video.srcObject = stream
  await video.play()
  overlay.width = video.videoWidth || 640
  overlay.height = video.videoHeight || 480
  return stream
}

// ---- Synthesis (demo-only, generates Chinese text summary + timeline) ----
class SynthesisEngine {
  constructor() { this.lastState = ''; this.stateStart = Date.now(); this.events = [] }
  synthesize(presence, attention, emotion) {
    let text = ''
    if (presence.count === 0) {
      text = '无人在屏幕前'
    } else if (presence.count > 1) {
      text = `${presence.count} 人在屏幕前`
    } else {
      const parts = []
      if (!presence.facing) {
        parts.push('没看屏幕')
      } else {
        if (attention.focus.level === 'high') {
          if (attention.gaze.region !== 'center' && attention.gaze.region !== 'unknown') {
            parts.push(`看着屏幕${attention.gaze.region}`)
          } else {
            parts.push('注视屏幕中')
          }
        } else if (attention.focus.level === 'medium') {
          parts.push('在看，有点走神')
        } else {
          parts.push('心不在焉')
        }
      }
      if (emotion.expression && emotion.expression !== '😐 平静' && emotion.expression !== 'neutral') {
        parts.push(emotion.expression)
      }
      text = parts.join('，') || '面对屏幕'
    }
    const stateKey = `${presence.count}-${presence.facing}-${attention.focus.level}`
    if (stateKey !== this.lastState) {
      const duration = ((Date.now() - this.stateStart) / 1000).toFixed(0)
      if (this.lastState && parseInt(duration) > 2) {
        this.events.push({
          time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
          text: `${this.lastState.startsWith('0') ? '离开' : '状态切换'} (${duration}s)`
        })
        if (this.events.length > 50) this.events.shift()
      }
      this.lastState = stateKey
      this.stateStart = Date.now()
    }
    return { text, events: this.events }
  }
}

// Expression label map (library returns English, dashboard expects Chinese with emoji)
const EXPR_MAP = {
  surprised: '😮 惊讶', talking: '🗣️ 说话', smiling: '😊 微笑',
  relaxed: '🙂 轻松', confused: '🤔 困惑', frowning: '🤨 皱眉',
  displeased: '😕 不悦', squinting: '😑 眯眼', neutral: '😐 平静',
}

// Adapt AgenticSense frame → Dashboard format
function adaptFrame(frame, synthesis) {
  const faceCount = frame.faceCount || 0
  const face = frame.faces?.[0]
  const interp = face?.interpretation

  const presence = {
    count: faceCount,
    facing: interp?.pose?.facing ?? false,
    distance: interp?.distance != null ? String(interp.distance) : '-',
  }

  const attention = {
    gaze: interp?.gaze ?? { region: '未知', looking: false },
    blinkRate: interp?.blinkRate ?? 0,
    focus: interp?.focus ?? { score: 0, level: '-' },
  }

  // Posture/tilt from head pose
  let posture = '正常'
  if (interp?.pose) {
    if (interp.pose.pitch < -0.15) posture = '前倾'
    else if (interp.pose.pitch > 0.15) posture = '后仰'
    else if (Math.abs(interp.pose.yaw) > 0.2) posture = '侧头'
  }
  let tilt = '正'
  if (interp?.pose?.roll != null) {
    const r = interp.pose.roll
    if (r > 8) tilt = '右倾'
    else if (r < -8) tilt = '左倾'
  }

  const exprKey = interp?.expression || 'neutral'
  const expression = EXPR_MAP[exprKey] || `😐 ${exprKey}`

  const emotion = { expression, posture, tilt }
  const synth = synthesis.synthesize(presence, attention, emotion)

  return { presence, attention, emotion, synthesis: synth, pose: interp?.pose, sense: frame }
}

async function init() {
  const startBtn = document.getElementById('start-btn')
  const startOverlay = document.getElementById('start-overlay')
  const app = document.getElementById('app')
  const startSelect = document.getElementById('camera-select')
  const hudSelect = document.getElementById('camera-switch')

  // Get camera list
  try {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true })
    tempStream.getTracks().forEach(t => t.stop())
    const cameras = await enumerateCameras()
    populateSelect(startSelect, cameras)
    populateSelect(hudSelect, cameras)
  } catch (e) {
    startSelect.innerHTML = '<option value="">默认摄像头</option>'
  }

  // Wait for start
  await new Promise(resolve => {
    startBtn.addEventListener('click', resolve, { once: true })
  })

  const selectedDeviceId = startSelect.value
  startOverlay.style.display = 'none'
  app.style.display = 'block'

  try {
    await startCamera(selectedDeviceId)

    if (hudSelect.value !== selectedDeviceId) {
      hudSelect.value = selectedDeviceId
    }

    // Init AgenticSense
    sense = new AgenticSense(video)
    await sense.init({
      wasmPath: '../mediapipe/',
      face: true,
      hands: true,
      pose: true,
      segment: true,
      objects: true,
    })

    const dashboard = new Dashboard()
    const synthesis = new SynthesisEngine()
    const ctx = overlay.getContext('2d')

    function loop() {
      const frame = sense.detect()
      if (frame) {
        // Draw overlay using raw MediaPipe results
        const raw = sense.rawResults
        drawOverlay(ctx, overlay, video, raw.face, raw.hand, raw.pose, raw.objects)

        // Update dashboard with adapted format
        const result = adaptFrame(frame, synthesis)
        dashboard.update(result)

        // Press 's' to dump sense frame
        if (window.__dumpSense) {
          console.log('SenseFrame:', JSON.stringify(frame, null, 2))
          window.__dumpSense = false
        }
      }
      requestAnimationFrame(loop)
    }
    requestAnimationFrame(loop)

    window.addEventListener('keydown', (e) => {
      if (e.key === 's') window.__dumpSense = true
    })

    hudSelect.addEventListener('change', async () => {
      try {
        await startCamera(hudSelect.value)
      } catch (e) {
        console.error('Camera switch failed:', e)
      }
    })

  } catch (e) {
    console.error('Init failed:', e)
    const synthEl = document.getElementById('synthesis-text')
    if (synthEl) synthEl.textContent = `错误: ${e.message}`
    if (statusDot) statusDot.style.background = 'var(--red)'
  }
}

init()
