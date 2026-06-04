"""Quick test: POST a WAV file to /predict endpoint"""
import urllib.request
import json

url = 'http://localhost:8000/predict'
boundary = 'testboundary123'

with open('test_tone.wav', 'rb') as f:
    wav_data = f.read()

parts = []
parts.append(b'--' + boundary.encode() + b'\r\n')
parts.append(b'Content-Disposition: form-data; name="file"; filename="test_tone.wav"\r\n')
parts.append(b'Content-Type: audio/wav\r\n\r\n')
parts.append(wav_data)
parts.append(b'\r\n--' + boundary.encode() + b'--\r\n')
body = b''.join(parts)

req = urllib.request.Request(url, data=body, method='POST')
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
req.add_header('Content-Length', str(len(body)))

try:
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode())
    print('SUCCESS!')
    print('Ensemble:', result['ensemble'])
    print('SVM:', result['svm'])
    print('GBM:', result['gbm'])
    print('RF:', result['rf'])
    print('Features:', result['features'])
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print('HTTP Error:', e.code, err)
except Exception as e:
    print('Error:', e)
