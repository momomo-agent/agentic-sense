"""SenseVoice server — loads model once, serves transcription via HTTP.

Usage:
  python sensevoice-server.py [--port 18906] [--host 0.0.0.0]

Endpoints:
  POST /transcribe  — send audio bytes, get {"text": "...", "time": 0.12}
  GET  /health      — {"ok": true, "model": "SenseVoiceSmall+VAD", "device": "..."}
"""
import os, sys, time, json, re, tempfile, argparse, platform
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['FUNASR_LOG_LEVEL'] = 'ERROR'

# ── Device detection ──────────────────────────────────────────────
def detect_device():
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    if torch.cuda.is_available():
        return 'cuda'  # NVIDIA GPU
    return 'cpu'

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='SenseVoice transcription server')
parser.add_argument('--port', type=int, default=18906)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

# ── Load models ───────────────────────────────────────────────────
print("Loading models...", flush=True)
t0 = time.time()
from funasr import AutoModel

# VAD — always CPU (tiny model, fast enough)
vad_model = AutoModel(model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', device='cpu', disable_update=True)
print(f"  VAD loaded: {time.time()-t0:.1f}s", flush=True)

# STT — use best available device
DEVICE = detect_device()
stt_model = AutoModel(model='iic/SenseVoiceSmall', device=DEVICE, disable_update=True)
print(f"  STT loaded ({DEVICE}): {time.time()-t0:.1f}s", flush=True)

# Warmup (compile GPU kernels on first run)
print("  Warming up...", flush=True)
warmup_path = os.path.join(tempfile.gettempdir(), 'sv-warmup.wav')
os.system(f'ffmpeg -y -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 "{warmup_path}" 2>/dev/null')
if os.path.exists(warmup_path):
    stt_model.generate(input=warmup_path, language='zh', use_itn=True)
    os.unlink(warmup_path)
print(f"  Ready in {time.time()-t0:.1f}s", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default logging

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != '/transcribe':
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', 0))
        audio = self.rfile.read(length)

        ct = self.headers.get('Content-Type', '')
        ext = 'webm' if 'webm' in ct else ('mp3' if 'mp3' in ct or 'mpeg' in ct else 'wav')

        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as f:
            f.write(audio)
            tmp = f.name

        wav_path = tmp
        if ext != 'wav':
            wav_path = tmp + '.wav'
            os.system(f'ffmpeg -y -i "{tmp}" -ar 16000 -ac 1 -c:a pcm_s16le "{wav_path}" 2>/dev/null')

        try:
            t1 = time.time()
            vad_res = vad_model.generate(input=wav_path)
            vad_time = time.time() - t1

            segments = vad_res[0].get('value', []) if vad_res else []
            segments = [s for s in segments if (s[1] - s[0]) > 300]

            if not segments:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({'text': '', 'vad': False, 'time': round(vad_time, 3)}).encode())
                return

            t2 = time.time()
            res = stt_model.generate(input=wav_path, language='zh', use_itn=True)
            stt_time = time.time() - t2

            text = ''
            if res and len(res) > 0:
                raw = res[0].get('text', '')
                text = re.sub(r'<\|[^|]*\|>', '', raw).strip()

            total = time.time() - t1
            print(f'[sensevoice] "{text}" ({len(audio)}B, vad:{vad_time:.2f}s stt:{stt_time:.2f}s total:{total:.2f}s)', flush=True)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({'text': text, 'vad': True, 'time': round(total, 2)}).encode())
        except Exception as e:
            print(f"[error] {e}", flush=True)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        finally:
            for p in set([tmp, wav_path]):
                try: os.unlink(p)
                except: pass

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'model': 'SenseVoiceSmall+VAD', 'device': DEVICE}).encode())
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    import socketserver
    socketserver.TCPServer.allow_reuse_address = True
    server = HTTPServer((args.host, args.port), Handler)
    server.allow_reuse_address = True
    print(f"\n🎤 SenseVoice server on http://{args.host}:{args.port}", flush=True)
    print(f"   Device: {DEVICE} | {platform.system()} {platform.machine()}", flush=True)
    print(f"   POST /transcribe — send audio, get text", flush=True)
    print(f"   GET  /health     — check status\n", flush=True)
    server.serve_forever()
