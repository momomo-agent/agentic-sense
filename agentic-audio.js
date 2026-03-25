/**
 * agentic-audio.js — Speech recognition + wake word detection for AI agents
 *
 * Uses Whisper WASM (Transformers.js) in a Web Worker for local speech-to-text
 * and Web Audio API for real-time volume + VAD.
 *
 * Wake word detection uses multi-modal fusion:
 * - With camera: silence gap + facing screen + wake word position
 * - Without camera: silence gap + wake word at sentence start
 *
 * Part of the agentic.js family.
 *
 * @version 0.1.0
 * @license MIT
 */
export class AgenticAudio {
  constructor(options = {}) {
    this.wakeWords = (options.wakeWords || ['hello', 'hey momo', 'momo'])
      .map(w => w.toLowerCase())
    this.lang = options.lang || 'zh-CN'
    this.workerPath = options.workerPath || './whisper-worker.js'

    // Callbacks
    this.onResult = null      // (text, isFinal, wakeJudgment)
    this.onVolumeChange = null // (volume 0-1)
    this.onWake = null         // (wakeWord, fullText, judgment)
    this.onModelStatus = null  // (status, message) model loading status

    this.worker = null
    this.audioCtx = null
    this._stopped = false
    this._supported = true
    this._modelReady = false

    // Audio capture state
    this.audioChunks = []       // Float32Array buffers
    this.silenceStart = 0
    this.isSpeaking = false
    this.vadThreshold = 0.01    // RMS threshold for VAD
    this.chunkDurationMs = 3000 // send to whisper every 3s while speaking
    this.lastChunkTime = 0

    // Wake judgment state
    this._lastSpeechTime = 0
    this._silenceThresholdMs = 1500
    this._facing = null
    this._hasCamera = false
  }

  get supported() { return this._supported }

  /** Update visual context from camera (call from main loop) */
  updateVisualContext(facing) {
    this._facing = facing
    this._hasCamera = true
  }

  /** Clear camera context (no camera available) */
  clearVisualContext() {
    this._facing = null
    this._hasCamera = false
  }

  async start() {
    this._stopped = false

    // 1. Start Worker and load Whisper model
    this.worker = new Worker(this.workerPath, { type: 'module' })
    this.worker.onmessage = (e) => this._handleWorkerMessage(e)
    this.worker.postMessage({ type: 'init' })

    // 2. Get microphone and set up audio capture
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      const source = this.audioCtx.createMediaStreamSource(stream)

      // ScriptProcessorNode for PCM capture + VAD
      const processor = this.audioCtx.createScriptProcessor(4096, 1, 1)
      source.connect(processor)
      processor.connect(this.audioCtx.destination)

      processor.onaudioprocess = (e) => {
        if (this._stopped) return

        const data = e.inputBuffer.getChannelData(0)
        const rms = this._computeRMS(data)

        if (this.onVolumeChange) this.onVolumeChange(rms)

        // VAD
        if (rms > this.vadThreshold) {
          if (!this.isSpeaking) {
            this.isSpeaking = true
            this.audioChunks = []
            this.lastChunkTime = Date.now()
          }
          this.silenceStart = 0
          this.audioChunks.push(new Float32Array(data))
        } else if (this.isSpeaking) {
          if (!this.silenceStart) this.silenceStart = Date.now()
          this.audioChunks.push(new Float32Array(data))

          // Silence > 800ms means utterance ended
          if (Date.now() - this.silenceStart > 800) {
            this.isSpeaking = false
            this._sendToWhisper(true)
          }
        }

        // Send interim results every chunkDurationMs while speaking
        if (this.isSpeaking && Date.now() - this.lastChunkTime > this.chunkDurationMs) {
          this._sendToWhisper(false)
        }
      }

      this._stream = stream
      this._processor = processor
      this._source = source
    } catch (e) {
      console.warn('Audio capture init failed:', e)
      this._supported = false
      return false
    }

    return true
  }

  _computeRMS(data) {
    let sum = 0
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i]
    }
    return Math.sqrt(sum / data.length)
  }

  _sendToWhisper(isFinal) {
    if (!this._modelReady || this.audioChunks.length === 0) return

    const totalLength = this.audioChunks.reduce((sum, c) => sum + c.length, 0)
    const merged = new Float32Array(totalLength)
    let offset = 0
    for (const chunk of this.audioChunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    this.worker.postMessage({ type: 'transcribe', audio: merged, isFinal }, [merged.buffer])

    if (isFinal) {
      this.audioChunks = []
    }
    this.lastChunkTime = Date.now()
  }

  _handleWorkerMessage(e) {
    const { type, status, message, text } = e.data

    if (type === 'status') {
      this._modelReady = (status === 'ready')
      if (this.onModelStatus) this.onModelStatus(status, message)
    }

    if (type === 'result') {
      const trimmed = (text || '').trim()
      if (!trimmed) return

      const lower = trimmed.toLowerCase()
      const judgment = this._judgeWake(lower, trimmed, true)

      this._lastSpeechTime = Date.now()

      if (this.onResult) {
        this.onResult(trimmed, true, judgment)
      }

      if (judgment.isWake && this.onWake) {
        this.onWake(judgment.wakeWord, trimmed, judgment)
      }
    }
  }

  /**
   * Multi-modal wake word judgment
   * Returns: { isWake, wakeWord, confidence, reason, signals }
   */
  _judgeWake(lower, originalText, isFinal) {
    const result = { isWake: false, wakeWord: null, confidence: 0, reason: '', signals: {} }

    let matchedWord = null
    let matchIdx = -1
    for (const w of this.wakeWords) {
      const idx = lower.indexOf(w)
      if (idx >= 0) {
        matchedWord = w
        matchIdx = idx
        break
      }
    }
    if (!matchedWord) return result

    result.wakeWord = matchedWord

    const now = Date.now()
    const silenceGap = now - this._lastSpeechTime
    const isAfterSilence = this._lastSpeechTime === 0 || silenceGap > this._silenceThresholdMs
    const isAtStart = matchIdx <= 2
    const isFacing = this._hasCamera ? this._facing === true : null

    result.signals = {
      silenceGap: isAfterSilence,
      atStart: isAtStart,
      facing: isFacing,
      hasCamera: this._hasCamera
    }

    let score = 0
    if (isAfterSilence) score += 0.4
    if (isAtStart) score += 0.35
    if (this._hasCamera && isFacing) score += 0.2

    result.confidence = Math.max(0, Math.min(1, score))

    if (result.confidence >= 0.5) {
      result.isWake = true
      const reasons = []
      if (isAfterSilence) reasons.push('silence gap')
      if (isAtStart) reasons.push('at start')
      if (isFacing === true) reasons.push('facing screen')
      result.reason = reasons.join(' + ')
    } else {
      const reasons = []
      if (!isAfterSilence) reasons.push('mid-speech')
      if (!isAtStart) reasons.push('wake word mid-sentence')
      if (isFacing === false) reasons.push('not facing')
      result.reason = reasons.join(' + ')
    }

    return result
  }

  stop() {
    this._stopped = true

    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }

    if (this._processor) {
      this._processor.disconnect()
      this._processor = null
    }

    if (this._source) {
      this._source.disconnect()
      this._source = null
    }

    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop())
      this._stream = null
    }

    if (this.audioCtx) {
      this.audioCtx.close().catch(() => {})
      this.audioCtx = null
    }
  }
}
