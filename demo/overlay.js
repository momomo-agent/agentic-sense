/**
 * Overlay drawing — extracted from engine for demo visualization
 * Takes raw MediaPipe results and draws landmarks on a canvas
 */

export function drawOverlay(ctx, canvas, video, faceResults, handResult, poseResult, objectResult) {
  // Match canvas size to displayed video size (cover fit)
  const rect = video.getBoundingClientRect()
  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width
    canvas.height = rect.height
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Compute cover transform
  const vw = video.videoWidth
  const vh = video.videoHeight
  const dw = rect.width
  const dh = rect.height

  const videoAspect = vw / vh
  const displayAspect = dw / dh

  let scale, offsetX, offsetY
  if (videoAspect > displayAspect) {
    scale = dh / vh; offsetX = (dw - vw * scale) / 2; offsetY = 0
  } else {
    scale = dw / vw; offsetX = 0; offsetY = (dh - vh * scale) / 2
  }

  const toX = (nx) => dw - (nx * vw * scale + offsetX)
  const toY = (ny) => ny * vh * scale + offsetY

  // ---- Face overlay ----
  if (faceResults?.faceLandmarks) {
    for (const landmarks of faceResults.faceLandmarks) {
      // Face oval
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.35)'
      ctx.lineWidth = 1.5

      const ovalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109, 10]

      ctx.beginPath()
      ctx.moveTo(toX(landmarks[ovalIndices[0]].x), toY(landmarks[ovalIndices[0]].y))
      for (let i = 1; i < ovalIndices.length; i++) {
        const p = landmarks[ovalIndices[i]]
        ctx.lineTo(toX(p.x), toY(p.y))
      }
      ctx.stroke()

      // Eye corners
      ctx.fillStyle = 'rgba(96, 165, 250, 0.7)'
      for (const i of [33, 133, 362, 263]) {
        const p = landmarks[i]
        ctx.beginPath()
        ctx.arc(toX(p.x), toY(p.y), 2.5, 0, Math.PI * 2)
        ctx.fill()
      }

      // Iris
      if (landmarks.length >= 478) {
        ctx.fillStyle = 'rgba(251, 191, 36, 0.85)'
        for (const i of [468, 473]) {
          const p = landmarks[i]
          ctx.beginPath()
          ctx.arc(toX(p.x), toY(p.y), 3.5, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // Nose tip
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'
      const nose = landmarks[1]
      ctx.beginPath()
      ctx.arc(toX(nose.x), toY(nose.y), 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // ---- Hand overlay ----
  if (handResult?.landmarks) {
    const HAND_CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],       // thumb
      [0,5],[5,6],[6,7],[7,8],       // index
      [0,9],[9,10],[10,11],[11,12],  // middle
      [0,13],[13,14],[14,15],[15,16],// ring
      [0,17],[17,18],[18,19],[19,20],// pinky
      [5,9],[9,13],[13,17]           // palm
    ]

    for (let h = 0; h < handResult.landmarks.length; h++) {
      const hand = handResult.landmarks[h]
      const color = h === 0 ? 'rgba(168, 85, 247, 0.8)' : 'rgba(236, 72, 153, 0.8)' // purple / pink

      // Connections
      ctx.strokeStyle = color.replace('0.8', '0.4')
      ctx.lineWidth = 1.5
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath()
        ctx.moveTo(toX(hand[a].x), toY(hand[a].y))
        ctx.lineTo(toX(hand[b].x), toY(hand[b].y))
        ctx.stroke()
      }

      // Joints
      ctx.fillStyle = color
      for (const pt of hand) {
        ctx.beginPath()
        ctx.arc(toX(pt.x), toY(pt.y), 2.5, 0, Math.PI * 2)
        ctx.fill()
      }

      // Gesture label
      if (handResult.gestures && handResult.gestures[h] && handResult.gestures[h].length > 0) {
        const gesture = handResult.gestures[h][0]
        if (gesture.categoryName !== 'None') {
          const wrist = hand[0]
          ctx.font = '14px "Space Mono", monospace'
          ctx.fillStyle = '#fff'
          ctx.fillText(gesture.categoryName, toX(wrist.x) + 10, toY(wrist.y) - 10)
        }
      }
    }
  }

  // ---- Pose overlay ----
  if (poseResult?.landmarks && poseResult.landmarks.length > 0) {
    const pose = poseResult.landmarks[0]
    const POSE_CONNECTIONS = [
      [11,12],                         // shoulders
      [11,13],[13,15],                 // left arm
      [12,14],[14,16],                 // right arm
      [11,23],[12,24],                 // torso
      [23,24],                         // hips
      [23,25],[25,27],                 // left leg
      [24,26],[26,28],                 // right leg
    ]

    // Connections
    ctx.strokeStyle = 'rgba(45, 212, 191, 0.4)'
    ctx.lineWidth = 2
    for (const [a, b] of POSE_CONNECTIONS) {
      if (pose[a] && pose[b]) {
        ctx.beginPath()
        ctx.moveTo(toX(pose[a].x), toY(pose[a].y))
        ctx.lineTo(toX(pose[b].x), toY(pose[b].y))
        ctx.stroke()
      }
    }

    // Joints
    ctx.fillStyle = 'rgba(45, 212, 191, 0.8)'
    for (let i = 11; i < Math.min(pose.length, 29); i++) {
      const pt = pose[i]
      if (pt) {
        ctx.beginPath()
        ctx.arc(toX(pt.x), toY(pt.y), 3, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  // ---- Object detection overlay ----
  if (objectResult?.detections) {
    for (const det of objectResult.detections) {
      const bb = det.boundingBox
      if (!bb) continue

      // BoundingBox is in pixel coords (not normalized), need to convert
      const x1 = toX((bb.originX + bb.width) / vw)
      const y1 = toY(bb.originY / vh)
      const x2 = toX(bb.originX / vw)
      const y2 = toY((bb.originY + bb.height) / vh)

      ctx.strokeStyle = 'rgba(251, 146, 60, 0.6)'  // orange
      ctx.lineWidth = 1.5
      ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1))

      // Label
      const cat = det.categories?.[0]
      if (cat) {
        const label = `${cat.categoryName} ${(cat.score * 100).toFixed(0)}%`
        ctx.font = '11px "Space Mono", monospace'
        ctx.fillStyle = 'rgba(251, 146, 60, 0.9)'
        ctx.fillText(label, Math.min(x1, x2), Math.min(y1, y2) - 4)
      }
    }
  }
}
