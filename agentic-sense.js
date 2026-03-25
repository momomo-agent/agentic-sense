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
