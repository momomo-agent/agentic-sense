/**
 * agentic-sense.js — Human perception for AI agents
 * 
 * Single-file library. Zero runtime dependencies.
 * MediaPipe WASM + models loaded at runtime by consumer.
 * 
 * Usage:
 *   import { AgenticSense } from './agentic-sense.js'
 *   const sense = new AgenticSense(videoElement)
 *   await sense.init({ wasmPath: './mediapipe/', face: true })
 *   const frame = sense.detect()
 * 
 * @version 0.1.0
 * @license MIT
 */

// ════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════

const IDX = {
  noseTip: 1, forehead: 10, chin: 152,
  leftFaceEdge: 234, rightFaceEdge: 454,
  leftEyeInner: 133, leftEyeOuter: 33,
  rightEyeInner: 362, rightEyeOuter: 263,
  leftEAR: [33, 160, 158, 133, 153, 144],
  rightEAR: [362, 385, 387, 263, 380, 373],
  leftIris: 468, rightIris: 473,
  upperLipCenter: 13, lowerLipCenter: 14,
  mouthLeft: 61, mouthRight: 291,
  faceOval: [10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,172,58,
    132,93,234,127,162,21,54,103,67,109,10]
}

const BS_MAP = {
  jawOpen:'jawOpen', mouthSmileLeft:'smileL', mouthSmileRight:'smileR',
  mouthFrownLeft:'frownL', mouthFrownRight:'frownR', mouthPucker:'pucker',
  browInnerUp:'browUp', browDownLeft:'browDownL', browDownRight:'browDownR',
  eyeSquintLeft:'squintL', eyeSquintRight:'squintR',
  eyeWideLeft:'eyeWideL', eyeWideRight:'eyeWideR',
  eyeBlinkLeft:'blinkL', eyeBlinkRight:'blinkR', cheekPuff:'cheekPuff',
  mouthClose:'mouthClose', mouthOpen:'mouthOpen',
  noseSneerLeft:'sneerL', noseSneerRight:'sneerR',
  jawLeft:'jawLeft', jawRight:'jawRight', mouthLeft:'mouthL', mouthRight:'mouthR',
  mouthShrugUpper:'shrugUpper', mouthShrugLower:'shrugLower',
  mouthRollUpper:'rollUpper', mouthRollLower:'rollLower',
  mouthFunnel:'funnel', mouthDimpleLeft:'dimpleL', mouthDimpleRight:'dimpleR',
  mouthStretchLeft:'stretchL', mouthStretchRight:'stretchR',
  mouthPressLeft:'pressL', mouthPressRight:'pressR',
  mouthLowerDownLeft:'lowerDownL', mouthLowerDownRight:'lowerDownR',
  mouthUpperUpLeft:'upperUpL', mouthUpperUpRight:'upperUpR',
}

const POSE_NAMES = {
  0:'nose', 11:'leftShoulder', 12:'rightShoulder',
  13:'leftElbow', 14:'rightElbow', 15:'leftWrist', 16:'rightWrist',
  23:'leftHip', 24:'rightHip', 25:'leftKnee', 26:'rightKnee',
  27:'leftAnkle', 28:'rightAnkle',
}

// ════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════

function round(v, d = 4) { const f = 10 ** d; return Math.round(v * f) / f }
function midpoint(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }
function computeEAR(pts) {
  if (pts.length < 6) return 1
  const [p1,p2,p3,p4,p5,p6] = pts
  const v1 = Math.hypot(p2.x-p6.x, p2.y-p6.y)
  const v2 = Math.hypot(p3.x-p5.x, p3.y-p5.y)
  const h = Math.hypot(p1.x-p4.x, p1.y-p4.y)
  return (v1 + v2) / (2 * h + 0.001)
}

class EMA {
  constructor(alpha = 0.3) { this.alpha = alpha; this.value = null }
  update(v) {
    if (this.value === null) { this.value = v; return v }
    this.value = this.value * (1 - this.alpha) + v * this.alpha
    return this.value
  }
  get() { return this.value }
}

// ════════════════════════════════════════════
// Feature extractors
// ════════════════════════════════════════════

class BlinkDetector {
  constructor() { this.history = []; this.wasOpen = true; this.threshold = 0.2 }
  update(ear) {
    const isOpen = ear > this.threshold
    if (this.wasOpen && !isOpen) {
      this.history.push(Date.now())
      this.history = this.history.filter(t => t > Date.now() - 60000)
    }
    this.wasOpen = isOpen
    return this.history.filter(t => t > Date.now() - 60000).length
  }
}

class HeadPoseEstimator {
  estimate(lm) {
    if (!lm || lm.length < 468) return { yaw:0, pitch:0, roll:0, facing:true }
    const nose = lm[1], left = lm[234], right = lm[454], top = lm[10], chin = lm[152]
    const fw = Math.abs(right.x - left.x), fh = Math.abs(chin.y - top.y)
    const cx = (left.x + right.x) / 2, cy = (top.y + chin.y) / 2
    const yaw = (nose.x - cx) / (fw + 0.001)
    const pitch = (nose.y - cy) / (fh + 0.001)
    const le = lm[33], re = lm[263]
    const roll = Math.atan2(re.y - le.y, re.x - le.x) * 180 / Math.PI
    return { yaw: round(yaw), pitch: round(pitch), roll: round(roll), facing: Math.abs(yaw) < 0.15 && Math.abs(pitch) < 0.2 }
  }
}

class GazeEstimator {
  estimate(lm) {
    if (!lm || lm.length < 478) return { region: 'unknown', looking: false, x: 0, y: 0 }
    const le33=lm[33],le133=lm[133],li=lm[468],re362=lm[362],re263=lm[263],ri=lm[473]
    const lw=Math.abs(le133.x-le33.x), rw=Math.abs(re263.x-re362.x)
    const gx=((li.x-le33.x)/(lw+.001)+(ri.x-re362.x)/(rw+.001))/2-.5
    const le159=lm[159],le145=lm[145],re386=lm[386],re374=lm[374]
    const lh=Math.abs(le145.y-le159.y), rh=Math.abs(re374.y-re386.y)
    const gy=((li.y-le159.y)/(lh+.001)-.5+(ri.y-re386.y)/(rh+.001)-.5)/2
    let region='center'
    if(gx<-.15) region='left'; else if(gx>.15) region='right'
    if(gy<-.15) region='up'+(region!=='center'?'-'+region:'')
    else if(gy>.15) region='down'+(region!=='center'?'-'+region:'')
    return { region, looking: Math.abs(gx)<.25 && Math.abs(gy)<.25, x: round(gx), y: round(gy) }
  }
}

class FocusScorer {
  constructor() { this.facingHistory=[]; this.bufferSize=90; this.lastGazeX=0; this.lastGazeY=0; this.gazeJitter=0 }
  update(facing, gaze, blinkRate) {
    this.facingHistory.push(facing ? 1 : 0)
    if (this.facingHistory.length > this.bufferSize) this.facingHistory.shift()
    if (gaze?.x !== undefined) {
      this.gazeJitter = this.gazeJitter * 0.95 + (Math.abs(gaze.x-this.lastGazeX)+Math.abs(gaze.y-this.lastGazeY)) * 0.05
      this.lastGazeX=gaze.x; this.lastGazeY=gaze.y
    }
    const facingRatio = this.facingHistory.reduce((a,b)=>a+b,0) / this.facingHistory.length
    let bf=1; if(blinkRate<5) bf=.75; if(blinkRate>28) bf=.65
    let gf=1; if(this.gazeJitter>.02) gf=.8; if(this.gazeJitter>.05) gf=.6
    const score = Math.round(facingRatio * 100 * bf * gf)
    return { score, level: score<35?'low':score<65?'medium':'high' }
  }
}

class ExpressionClassifier {
  classify(bs) {
    if (!bs) return 'neutral'
    const jawOpen=bs.jawOpen||0, smileL=bs.smileL||0, smileR=bs.smileR||0
    const browUp=bs.browUp||0, browDownL=bs.browDownL||0, browDownR=bs.browDownR||0
    const eyeWideL=bs.eyeWideL||0, eyeWideR=bs.eyeWideR||0
    const frownL=bs.frownL||0, frownR=bs.frownR||0
    const squintL=bs.squintL||0, squintR=bs.squintR||0
    const smile=(smileL+smileR)/2, eyeWide=(eyeWideL+eyeWideR)/2, browDown=(browDownL+browDownR)/2
    if(jawOpen>.6 && eyeWide>.3) return 'surprised'
    if(jawOpen>.5) return 'talking'
    if(smile>.5) return 'smiling'
    if(smile>.25 && jawOpen<.3) return 'relaxed'
    if(browUp>.5 && browDown<.2) return 'confused'
    if(browDown>.4) return 'frowning'
    if((frownL+frownR)/2>.35) return 'displeased'
    if((squintL+squintR)/2>.6) return 'squinting'
    return 'neutral'
  }
}

class CustomGestureDetector {
  constructor() {
    this.pinchThreshold = 0.3 // normalized by palm width
  }

  isFingerExtended(landmarks, tipIdx, pipIdx) {
    return landmarks[tipIdx].y < landmarks[pipIdx].y
  }

  isThumbExtended(landmarks) {
    const tip = landmarks[4]
    const ip = landmarks[3]
    const mcp = landmarks[2]
    return Math.abs(tip.x - mcp.x) > Math.abs(ip.x - mcp.x)
  }

  pinchDistance(landmarks, idxA, idxB) {
    const a = landmarks[idxA]
    const b = landmarks[idxB]
    const palmWidth = Math.hypot(
      landmarks[5].x - landmarks[17].x,
      landmarks[5].y - landmarks[17].y
    )
    return Math.hypot(a.x - b.x, a.y - b.y) / (palmWidth + 0.001)
  }

  detect(landmarks) {
    if (!landmarks || landmarks.length < 21) return null

    const thumb = this.isThumbExtended(landmarks)
    const index = this.isFingerExtended(landmarks, 8, 6)
    const middle = this.isFingerExtended(landmarks, 12, 10)
    const ring = this.isFingerExtended(landmarks, 16, 14)
    const pinky = this.isFingerExtended(landmarks, 20, 18)

    const thumbIndexDist = this.pinchDistance(landmarks, 4, 8)
    const isPinching = thumbIndexDist < this.pinchThreshold

    if (isPinching && middle && ring && pinky) {
      return { name: 'OK', confidence: 0.8, emoji: '👌' }
    }
    if (isPinching && !middle && !ring && !pinky) {
      return { name: 'Pinch', confidence: 0.8, emoji: '🤏' }
    }
    if (index && pinky && !middle && !ring) {
      return { name: 'Rock', confidence: 0.75, emoji: '🤟' }
    }
    if (index && !middle && !ring && !pinky && !thumb) {
      return { name: 'One', confidence: 0.7, emoji: '☝️' }
    }
    if (index && middle && !ring && !pinky) {
      return { name: 'Peace', confidence: 0.75, emoji: '✌️' }
    }

    return null
  }
}

class ActionDetector {
  constructor() {
    this.bufferSize = 30
    this.handHistory = {}
    this.cooldowns = {}
    this.cooldownMs = 500
  }

  addFrame(handIdx, wrist, indexTip, palmCenter, timestamp) {
    if (!this.handHistory[handIdx]) {
      this.handHistory[handIdx] = []
    }
    const buf = this.handHistory[handIdx]
    buf.push({ wrist, indexTip, palmCenter, timestamp })
    if (buf.length > this.bufferSize) buf.shift()
  }

  canFire(action) {
    const last = this.cooldowns[action] || 0
    return (performance.now() - last) > this.cooldownMs
  }

  fire(action) {
    this.cooldowns[action] = performance.now()
  }

  detect(handIdx) {
    const buf = this.handHistory[handIdx]
    if (!buf || buf.length < 5) return []

    const actions = []
    const now = buf[buf.length - 1].timestamp

    // Wave — wrist x reversals >= 2 in last 1s
    if (this.canFire('Wave')) {
      const recent = buf.filter(f => (now - f.timestamp) < 1000)
      if (recent.length >= 5) {
        let reversals = 0
        let lastDir = 0
        for (let i = 1; i < recent.length; i++) {
          const dx = recent[i].wrist.x - recent[i - 1].wrist.x
          if (Math.abs(dx) > 0.005) {
            const dir = dx > 0 ? 1 : -1
            if (lastDir !== 0 && dir !== lastDir) reversals++
            lastDir = dir
          }
        }
        const totalDx = recent.reduce((sum, f, i) => {
          if (i === 0) return 0
          return sum + Math.abs(f.wrist.x - recent[i - 1].wrist.x)
        }, 0)
        if (reversals >= 2 && totalDx > 0.08) {
          actions.push({ name: 'Wave', emoji: '👋', confidence: 0.7 })
          this.fire('Wave')
        }
      }
    }

    // Tap — index z drops then rebounds
    if (this.canFire('Tap')) {
      const last5 = buf.slice(-5)
      if (last5.length >= 5) {
        const zVals = last5.map(f => f.indexTip.z)
        const minZ = Math.min(...zVals)
        const startZ = zVals[0]
        const endZ = zVals[zVals.length - 1]
        if (startZ - minZ > 0.03 && endZ - minZ > 0.02) {
          actions.push({ name: 'Tap', emoji: '☝️', confidence: 0.65 })
          this.fire('Tap')
        }
      }
    }

    // Push Down — palm y increases rapidly
    if (this.canFire('Push Down')) {
      const recent = buf.slice(-8)
      if (recent.length >= 6) {
        const dy = recent[recent.length - 1].palmCenter.y - recent[0].palmCenter.y
        const dt = recent[recent.length - 1].timestamp - recent[0].timestamp
        if (dy > 0.08 && dt < 500 && dt > 0) {
          actions.push({ name: 'Push Down', emoji: '🫳', confidence: 0.7 })
          this.fire('Push Down')
        }
      }
    }

    // Lift Up — palm y decreases rapidly
    if (this.canFire('Lift Up')) {
      const recent = buf.slice(-8)
      if (recent.length >= 6) {
        const dy = recent[recent.length - 1].palmCenter.y - recent[0].palmCenter.y
        const dt = recent[recent.length - 1].timestamp - recent[0].timestamp
        if (dy < -0.08 && dt < 500 && dt > 0) {
          actions.push({ name: 'Lift Up', emoji: '🫴', confidence: 0.7 })
          this.fire('Lift Up')
        }
      }
    }

    // Circle — index tip angle accumulates >= 360deg
    if (this.canFire('Circle')) {
      const recent = buf.filter(f => (now - f.timestamp) < 1500)
      if (recent.length >= 10) {
        const cx = recent.reduce((s, f) => s + f.indexTip.x, 0) / recent.length
        const cy = recent.reduce((s, f) => s + f.indexTip.y, 0) / recent.length
        let totalAngle = 0
        for (let i = 1; i < recent.length; i++) {
          const a1 = Math.atan2(recent[i - 1].indexTip.y - cy, recent[i - 1].indexTip.x - cx)
          const a2 = Math.atan2(recent[i].indexTip.y - cy, recent[i].indexTip.x - cx)
          let da = a2 - a1
          if (da > Math.PI) da -= 2 * Math.PI
          if (da < -Math.PI) da += 2 * Math.PI
          totalAngle += da
        }
        if (Math.abs(totalAngle) >= 2 * Math.PI) {
          actions.push({ name: 'Circle', emoji: '🔄', confidence: 0.65 })
          this.fire('Circle')
        }
      }
    }

    return actions
  }

  reset() {
    this.handHistory = {}
  }
}

// ════════════════════════════════════════════
// Frame extraction (from MediaPipe results → structured data)
// ════════════════════════════════════════════

function extractFace(landmarks, blendshapeCategories) {
  const lm = landmarks, hasIris = lm.length >= 478
  const nose=lm[1], forehead=lm[10], chin=lm[152], le=lm[234], re=lm[454]
  const fw=Math.abs(re.x-le.x), fh=Math.abs(chin.y-forehead.y)
  const cx=(le.x+re.x)/2, cy=(forehead.y+chin.y)/2
  const yaw=(nose.x-cx)/(fw+.001), pitch=(nose.y-cy)/(fh+.001)
  const leye=lm[33], reye=lm[263]
  const roll=Math.atan2(reye.y-leye.y, reye.x-leye.x)*180/Math.PI

  const leftEAR=computeEAR(IDX.leftEAR.map(i=>lm[i]))
  const rightEAR=computeEAR(IDX.rightEAR.map(i=>lm[i]))
  const leftCenter=midpoint(lm[133],lm[33]), rightCenter=midpoint(lm[362],lm[263])
  const ipd=Math.hypot(rightCenter.x-leftCenter.x, rightCenter.y-leftCenter.y)
  let iris=null
  if(hasIris){
    const li=lm[468],ri=lm[473]
    const ls=Math.abs(lm[33].x-lm[133].x), rs=Math.abs(lm[263].x-lm[362].x)
    iris={left:{x:li.x,y:li.y,ratioX:(li.x-lm[133].x)/(ls+.001)},right:{x:ri.x,y:ri.y,ratioX:(ri.x-lm[362].x)/(rs+.001)}}
  }

  const upper=lm[13],lower=lm[14],ml=lm[61],mr=lm[291]
  const mOpen=Math.abs(upper.y-lower.y), mWidth=Math.abs(ml.x-mr.x)

  const blendshapes={}, rawBlendshapes={}
  if(blendshapeCategories){
    for(const cat of blendshapeCategories){
      rawBlendshapes[cat.categoryName]=cat.score
      const key=BS_MAP[cat.categoryName]
      if(key) blendshapes[key]=round(cat.score,3)
    }
  }

  return {
    head: { position:{x:nose.x,y:nose.y,z:nose.z}, yaw:round(yaw), pitch:round(pitch), roll:round(roll), faceWidth:round(fw), faceHeight:round(fh), facing:Math.abs(yaw)<.15&&Math.abs(pitch)<.2 },
    eyes: { left:{center:leftCenter,ear:round(leftEAR)}, right:{center:rightCenter,ear:round(rightEAR)}, avgEAR:round((leftEAR+rightEAR)/2), ipd:round(ipd), iris },
    mouth: { openness:round(mOpen), width:round(mWidth), ratio:round(mOpen/(mWidth+.001)), center:midpoint(upper,lower) },
    blendshapes, rawBlendshapes, landmarkCount: lm.length,
  }
}

function extractHands(handResult, handLmResult) {
  const hands = []
  const src = handResult?.landmarks ? handResult : handLmResult
  if (!src?.landmarks) return hands
  for (let h = 0; h < src.landmarks.length; h++) {
    const lm = src.landmarks[h]
    const handedness = src.handednesses?.[h]?.[0]
    const gesture = handResult?.gestures?.[h]?.[0]
    const fingers = { thumb:lm[4].y<lm[3].y, index:lm[8].y<lm[6].y, middle:lm[12].y<lm[10].y, ring:lm[16].y<lm[14].y, pinky:lm[20].y<lm[18].y }
    hands.push({
      side: handedness?.categoryName||'unknown', confidence: round(handedness?.score||0),
      gesture: gesture?.categoryName!=='None'?gesture?.categoryName:null,
      gestureConfidence: gesture?.categoryName!=='None'?round(gesture?.score||0):null,
      wrist:{x:round(lm[0].x),y:round(lm[0].y),z:round(lm[0].z)},
      fingers, extendedCount: Object.values(fingers).filter(Boolean).length,
      landmarks: lm.map(p=>({x:round(p.x),y:round(p.y),z:round(p.z)})),
    })
  }
  return hands
}

function extractBody(poseResult) {
  if (!poseResult?.landmarks?.length) return null
  const pose = poseResult.landmarks[0]
  const joints = {}
  for (const [idx,name] of Object.entries(POSE_NAMES)) {
    const pt=pose[parseInt(idx)]
    if(pt) joints[name]={x:round(pt.x),y:round(pt.y),z:round(pt.z),visibility:round(pt.visibility||0)}
  }
  const sw=joints.leftShoulder&&joints.rightShoulder?round(Math.abs(joints.rightShoulder.x-joints.leftShoulder.x)):null
  const smy=joints.leftShoulder&&joints.rightShoulder?(joints.leftShoulder.y+joints.rightShoulder.y)/2:null
  const hmy=joints.leftHip&&joints.rightHip?(joints.leftHip.y+joints.rightHip.y)/2:null
  return { joints, shoulderWidth:sw, torsoLength:smy!==null&&hmy!==null?round(Math.abs(hmy-smy)):null, landmarkCount:pose.length }
}

function extractObjects(objectResult) {
  if (!objectResult?.detections) return []
  return objectResult.detections.map(det=>{
    const cat=det.categories?.[0], bb=det.boundingBox
    return cat&&bb?{label:cat.categoryName,confidence:round(cat.score),box:{x:round(bb.originX),y:round(bb.originY),width:round(bb.width),height:round(bb.height)}}:null
  }).filter(Boolean)
}

function extractFrame(faceResult, handResult, poseResult, segResult, objectResult, faceDetResult, handLmResult) {
  const faces = []
  const faceCount = faceResult?.faceLandmarks?.length || 0
  for (let i = 0; i < faceCount; i++) {
    faces.push(extractFace(faceResult.faceLandmarks[i], faceResult.faceBlendshapes?.[i]?.categories))
  }
  const hands = extractHands(handResult, handLmResult)
  const body = extractBody(poseResult)
  const objects = extractObjects(objectResult)

  let segmentation = null
  if (segResult?.categoryMask) {
    const data = segResult.categoryMask.getAsUint8Array()
    let pp = 0; for (let i = 0; i < data.length; i++) if (data[i] > 0) pp++
    segmentation = { personRatio: round(pp/data.length), width: segResult.categoryMask.width, height: segResult.categoryMask.height }
  }

  return { timestamp: performance.now(), faceCount, faces, handCount: hands.length, hands, body, segmentation, objectCount: objects.length, objects }
}

// ════════════════════════════════════════════
// AgenticSense — main class
// ════════════════════════════════════════════

export class AgenticSense {
  static get VERSION() { return '0.1.0' }

  constructor(videoElement, options = {}) {
    this.video = videoElement
    this.options = options
    this._vision = null
    this._faceLandmarker = null
    this._gestureRecognizer = null
    this._handLandmarker = null
    this._poseLandmarker = null
    this._imageSegmenter = null
    this._objectDetector = null

    // Feature extractors
    this._blink = new BlinkDetector()
    this._headPose = new HeadPoseEstimator()
    this._gaze = new GazeEstimator()
    this._focus = new FocusScorer()
    this._expression = new ExpressionClassifier()
    this._customGesture = new CustomGestureDetector()
    this._actionDetector = new ActionDetector()
    this._yawEMA = new EMA(0.2)
    this._pitchEMA = new EMA(0.2)
    this._distEMA = new EMA(0.15)

    this._frameCount = 0
    this._lastResult = null
    this._lastFace = null
    this._lastHand = null
    this._lastHandLm = null
    this._lastPose = null
    this._lastSeg = null
    this._lastObject = null
  }

  /**
   * Initialize MediaPipe models
   * @param {object} opts - { wasmPath, face, hands, pose, segment, objects }
   */
  async init(opts = {}) {
    const { wasmPath = './mediapipe/', face = true, hands = false, pose = false, segment = false, objects = false } = opts

    const vision = await import(wasmPath + 'vision_bundle.mjs')
    this._vision = vision
    const { FilesetResolver } = vision
    const wasm = await FilesetResolver.forVisionTasks(wasmPath)

    const inits = []

    if (face) {
      inits.push((async () => {
        this._faceLandmarker = await vision.FaceLandmarker.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: wasmPath + 'face_landmarker.task', delegate: 'GPU' },
          runningMode: 'VIDEO', numFaces: 3, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: false
        })
      })())
    }

    if (hands) {
      inits.push((async () => {
        if (vision.GestureRecognizer) {
          try {
            this._gestureRecognizer = await vision.GestureRecognizer.createFromOptions(wasm, {
              baseOptions: { modelAssetPath: wasmPath + 'gesture_recognizer.task', delegate: 'GPU' },
              runningMode: 'VIDEO', numHands: 2
            })
            return
          } catch(e) { /* fallback to HandLandmarker */ }
        }
        if (vision.HandLandmarker) {
          this._handLandmarker = await vision.HandLandmarker.createFromOptions(wasm, {
            baseOptions: { modelAssetPath: wasmPath + 'hand_landmarker.task', delegate: 'GPU' },
            runningMode: 'VIDEO', numHands: 2
          })
        }
      })())
    }

    if (pose && vision.PoseLandmarker) {
      inits.push((async () => {
        this._poseLandmarker = await vision.PoseLandmarker.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: wasmPath + 'pose_landmarker_lite.task', delegate: 'GPU' },
          runningMode: 'VIDEO', numPoses: 1
        })
      })())
    }

    if (segment && vision.ImageSegmenter) {
      inits.push((async () => {
        this._imageSegmenter = await vision.ImageSegmenter.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: wasmPath + 'selfie_segmenter.tflite', delegate: 'GPU' },
          runningMode: 'VIDEO', outputCategoryMask: true, outputConfidenceMasks: false
        })
      })())
    }

    if (objects && vision.ObjectDetector) {
      inits.push((async () => {
        this._objectDetector = await vision.ObjectDetector.createFromOptions(wasm, {
          baseOptions: { modelAssetPath: wasmPath + 'efficientdet_lite0.tflite', delegate: 'GPU' },
          runningMode: 'VIDEO', maxResults: 10, scoreThreshold: 0.3
        })
      })())
    }

    await Promise.all(inits)
  }

  /**
   * Detect one frame. Call this in requestAnimationFrame.
   * @returns {SenseFrame|null}
   */
  detect() {
    if (this.video.readyState < 2) return this._lastResult
    this._frameCount++
    const now = performance.now()

    // Face landmarks (every frame)
    let faceResults = null
    if (this._faceLandmarker) {
      try { faceResults = this._faceLandmarker.detectForVideo(this.video, now); this._lastFace = faceResults } catch(e) { return this._lastResult }
    }

    // Hands (every frame)
    if (this._gestureRecognizer) {
      try { this._lastHand = this._gestureRecognizer.recognizeForVideo(this.video, now) } catch(e) {}
    } else if (this._handLandmarker) {
      try { this._lastHandLm = this._handLandmarker.detectForVideo(this.video, now) } catch(e) {}
    }

    // Pose (every 2nd frame)
    if (this._poseLandmarker && this._frameCount % 2 === 0) {
      try { this._lastPose = this._poseLandmarker.detectForVideo(this.video, now) } catch(e) {}
    }

    // Segmentation (every 3rd frame)
    if (this._imageSegmenter && this._frameCount % 3 === 0) {
      try { this._lastSeg = this._imageSegmenter.segmentForVideo(this.video, now) } catch(e) {}
    }

    // Objects (every 5th frame)
    if (this._objectDetector && this._frameCount % 5 === 0) {
      try { this._lastObject = this._objectDetector.detectForVideo(this.video, now) } catch(e) {}
    }

    // Build raw sense frame
    const raw = extractFrame(faceResults, this._lastHand, this._lastPose, this._lastSeg, this._lastObject, null, this._lastHandLm)

    // Enrich hands with custom gestures and actions
    const customGestures = []
    const actions = []
    for (let h = 0; h < raw.hands.length; h++) {
      const hand = raw.hands[h]
      // Custom gesture: use if GestureRecognizer returned None
      const cg = this._customGesture.detect(hand.landmarks)
      if (cg) {
        if (!hand.gesture) {
          hand.gesture = cg.name
          hand.gestureConfidence = cg.confidence
        }
        customGestures.push({ hand: h, ...cg })
      }

      // Action detection
      const lm = hand.landmarks
      const wrist = lm[0]
      const indexTip = lm[8]
      const palmCenter = { x: (lm[0].x + lm[5].x + lm[17].x) / 3, y: (lm[0].y + lm[5].y + lm[17].y) / 3, z: (lm[0].z + lm[5].z + lm[17].z) / 3 }
      this._actionDetector.addFrame(h, wrist, indexTip, palmCenter, now)
      const handActions = this._actionDetector.detect(h)
      for (const a of handActions) {
        actions.push({ hand: h, ...a })
      }
    }
    raw.customGestures = customGestures
    raw.actions = actions

    // Enrich primary face with interpreted data
    if (raw.faceCount > 0) {
      const face = raw.faces[0]
      const pose = this._headPose.estimate(faceResults?.faceLandmarks?.[0])
      pose.yaw = this._yawEMA.update(pose.yaw)
      pose.pitch = this._pitchEMA.update(pose.pitch)
      pose.facing = Math.abs(pose.yaw) < 0.15 && Math.abs(pose.pitch) < 0.2

      const gaze = this._gaze.estimate(faceResults?.faceLandmarks?.[0])
      const blinkRate = this._blink.update(face.eyes.avgEAR)
      const focus = this._focus.update(pose.facing, gaze, blinkRate)
      const expression = this._expression.classify(face.blendshapes)

      const lm = faceResults.faceLandmarks[0]
      const le = lm[33], re = lm[263]
      const eyeDist = Math.hypot(re.x-le.x, re.y-le.y)
      const distance = this._distEMA.update(0.12 / (eyeDist + 0.001))

      face.interpretation = { pose, gaze, blinkRate, focus, expression, distance: round(distance, 1) }
    }

    this._lastResult = raw
    return raw
  }

  /**
   * Get raw MediaPipe results for overlay drawing (demo use)
   */
  get rawResults() {
    return {
      face: this._lastFace,
      hand: this._lastHand,
      handLm: this._lastHandLm,
      pose: this._lastPose,
      seg: this._lastSeg,
      objects: this._lastObject,
    }
  }

  destroy() {
    this._faceLandmarker?.close()
    this._gestureRecognizer?.close()
    this._handLandmarker?.close()
    this._poseLandmarker?.close()
    this._imageSegmenter?.close()
    this._objectDetector?.close()
    this._lastResult = null
  }
}

export { IDX, extractFrame }
