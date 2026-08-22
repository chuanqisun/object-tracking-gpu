# nms=False end2end=False non-exhaustive fp16 MLIR=0

```txt
Decode: 1.19ms | Prep: 0.59ms | Inf: 10.24ms | Post: 0.08ms | Track: 0.00ms | Total: 12.11ms (82.6 FPS)
Decode: 0.91ms | Prep: 0.60ms | Inf: 10.43ms | Post: 0.08ms | Track: 0.00ms | Total: 12.03ms (83.2 FPS)
Decode: 1.19ms | Prep: 1.15ms | Inf: 17.53ms | Post: 3.37ms | Track: 0.17ms | Total: 23.40ms (42.7 FPS)
Decode: 1.48ms | Prep: 1.21ms | Inf: 23.05ms | Post: 4.55ms | Track: 0.30ms | Total: 30.59ms (32.7 FPS)
Decode: 1.36ms | Prep: 1.16ms | Inf: 23.04ms | Post: 6.07ms | Track: 0.26ms | Total: 31.89ms (31.4 FPS)
Decode: 0.89ms | Prep: 0.62ms | Inf: 11.43ms | Post: 0.08ms | Track: 0.00ms | Total: 13.03ms (76.8 FPS)
Decode: 0.87ms | Prep: 0.63ms | Inf: 10.80ms | Post: 0.09ms | Track: 0.00ms | Total: 12.39ms (80.7 FPS)
Decode: 0.99ms | Prep: 0.61ms | Inf: 10.45ms | Post: 0.09ms | Track: 0.00ms | Total: 12.13ms (82.4 FPS)
```

# nms=False end2end=True non-exhaustive fp16 MLIR=1

```txt
Decode: 0.85ms | Prep: 0.65ms | Inf: 8.61ms | Post: 0.08ms | Track: 0.00ms | Total: 10.19ms (98.1 FPS)
Decode: 1.11ms | Prep: 0.62ms | Inf: 9.95ms | Post: 0.08ms | Track: 0.00ms | Total: 11.77ms (85.0 FPS)
Decode: 0.82ms | Prep: 0.60ms | Inf: 10.01ms | Post: 0.08ms | Track: 0.00ms | Total: 11.50ms (86.9 FPS)
Decode: 0.83ms | Prep: 0.57ms | Inf: 9.59ms | Post: 0.08ms | Track: 0.00ms | Total: 11.08ms (90.3 FPS)
Decode: 0.77ms | Prep: 0.59ms | Inf: 10.08ms | Post: 0.08ms | Track: 0.00ms | Total: 11.53ms (86.8 FPS)
Decode: 0.76ms | Prep: 0.60ms | Inf: 10.03ms | Post: 0.08ms | Track: 0.00ms | Total: 11.48ms (87.1 FPS)
Decode: 0.95ms | Prep: 0.76ms | Inf: 16.96ms | Post: 1.12ms | Track: 0.55ms | Total: 20.35ms (49.1 FPS)
Decode: 1.47ms | Prep: 1.09ms | Inf: 22.52ms | Post: 1.41ms | Track: 1.59ms | Total: 28.09ms (35.6 FPS)
Decode: 1.36ms | Prep: 1.02ms | Inf: 22.38ms | Post: 1.12ms | Track: 1.40ms | Total: 27.27ms (36.7 FPS)
Decode: 1.34ms | Prep: 0.90ms | Inf: 22.29ms | Post: 1.12ms | Track: 1.29ms | Total: 26.95ms (37.1 FPS)
Decode: 1.35ms | Prep: 0.89ms | Inf: 22.24ms | Post: 1.05ms | Track: 1.28ms | Total: 26.81ms (37.3 FPS)
Decode: 1.49ms | Prep: 0.89ms | Inf: 22.23ms | Post: 1.04ms | Track: 1.27ms | Total: 26.92ms (37.1 FPS)
Decode: 1.28ms | Prep: 0.89ms | Inf: 22.11ms | Post: 1.01ms | Track: 1.22ms | Total: 26.51ms (37.7 FPS)
Decode: 1.18ms | Prep: 0.88ms | Inf: 17.06ms | Post: 0.72ms | Track: 0.40ms | Total: 20.23ms (49.4 FPS)
Decode: 1.20ms | Prep: 0.60ms | Inf: 9.72ms | Post: 0.08ms | Track: 0.00ms | Total: 11.60ms (86.2 FPS)
Decode: 1.02ms | Prep: 0.61ms | Inf: 9.93ms | Post: 0.08ms | Track: 0.00ms | Total: 11.65ms (85.8 FPS)
Decode: 0.85ms | Prep: 0.61ms | Inf: 9.98ms | Post: 0.08ms | Track: 0.00ms | Total: 11.52ms (86.8 FPS)
Decode: 0.91ms | Prep: 0.62ms | Inf: 10.19ms | Post: 0.08ms | Track: 0.00ms | Total: 11.80ms (84.8 FPS)
```

# nms=False end2end=True non-exhaustive fp16 MLIR=1 max_det=10

```txt
Decode: 0.83ms | Prep: 0.61ms | Inf: 9.52ms | Post: 0.08ms | Track: 0.00ms | Total: 11.04ms (90.6 FPS)
Decode: 0.93ms | Prep: 0.69ms | Inf: 9.25ms | Post: 0.09ms | Track: 0.00ms | Total: 10.95ms (91.3 FPS)
Decode: 0.74ms | Prep: 0.58ms | Inf: 9.70ms | Post: 0.08ms | Track: 0.00ms | Total: 11.10ms (90.1 FPS)
Decode: 0.81ms | Prep: 0.60ms | Inf: 9.44ms | Post: 0.08ms | Track: 0.00ms | Total: 10.93ms (91.5 FPS)
Decode: 0.96ms | Prep: 0.60ms | Inf: 9.58ms | Post: 0.08ms | Track: 0.00ms | Total: 11.21ms (89.2 FPS)
Decode: 1.06ms | Prep: 0.65ms | Inf: 9.56ms | Post: 0.08ms | Track: 0.00ms | Total: 11.36ms (88.1 FPS)
Decode: 1.13ms | Prep: 0.90ms | Inf: 17.27ms | Post: 1.84ms | Track: 0.88ms | Total: 22.02ms (45.4 FPS)
Decode: 1.16ms | Prep: 0.92ms | Inf: 22.91ms | Post: 1.50ms | Track: 1.11ms | Total: 27.59ms (36.2 FPS)
Decode: 1.17ms | Prep: 0.87ms | Inf: 22.08ms | Post: 0.91ms | Track: 1.15ms | Total: 26.17ms (38.2 FPS)
Decode: 1.12ms | Prep: 0.94ms | Inf: 22.28ms | Post: 1.21ms | Track: 1.11ms | Total: 26.66ms (37.5 FPS)
Decode: 1.19ms | Prep: 0.92ms | Inf: 22.62ms | Post: 1.41ms | Track: 0.95ms | Total: 27.09ms (36.9 FPS)
Decode: 1.07ms | Prep: 0.57ms | Inf: 9.99ms | Post: 0.07ms | Track: 0.00ms | Total: 11.70ms (85.5 FPS)
Decode: 0.77ms | Prep: 0.60ms | Inf: 9.63ms | Post: 0.08ms | Track: 0.00ms | Total: 11.07ms (90.3 FPS)
Decode: 0.92ms | Prep: 0.64ms | Inf: 10.48ms | Post: 0.09ms | Track: 0.00ms | Total: 12.13ms (82.4 FPS)
Decode: 0.90ms | Prep: 0.67ms | Inf: 10.07ms | Post: 0.09ms | Track: 0.00ms | Total: 11.72ms (85.3 FPS)
```

# nms=False end2end=True exhaustive fp16 MLIR=1 removed tracker and mask handling

```txt
Decode: 0.75ms | Prep: 0.59ms | Inf: 10.38ms | Post: 0.17ms | Total: 11.90ms (84.0 FPS)
Decode: 0.76ms | Prep: 0.57ms | Inf: 10.16ms | Post: 0.17ms | Total: 11.68ms (85.6 FPS)
Decode: 0.81ms | Prep: 0.57ms | Inf: 9.87ms | Post: 0.18ms | Total: 11.45ms (87.4 FPS)
Decode: 0.78ms | Prep: 0.58ms | Inf: 9.97ms | Post: 0.08ms | Total: 11.40ms (87.7 FPS)
Decode: 0.78ms | Prep: 0.57ms | Inf: 10.05ms | Post: 0.17ms | Total: 11.58ms (86.3 FPS)
Decode: 0.89ms | Prep: 0.56ms | Inf: 10.03ms | Post: 0.17ms | Total: 11.66ms (85.8 FPS)
Decode: 0.88ms | Prep: 0.58ms | Inf: 9.86ms | Post: 0.18ms | Total: 11.51ms (86.9 FPS)
Decode: 0.75ms | Prep: 0.58ms | Inf: 10.21ms | Post: 0.18ms | Total: 11.74ms (85.2 FPS)
Decode: 0.80ms | Prep: 0.58ms | Inf: 10.11ms | Post: 0.17ms | Total: 11.68ms (85.6 FPS)
Decode: 1.07ms | Prep: 0.58ms | Inf: 9.78ms | Post: 0.14ms | Total: 11.58ms (86.3 FPS)
Decode: 0.89ms | Prep: 0.62ms | Inf: 9.90ms | Post: 0.13ms | Total: 11.54ms (86.6 FPS)
Decode: 0.76ms | Prep: 0.57ms | Inf: 9.84ms | Post: 0.08ms | Total: 11.24ms (88.9 FPS)
Decode: 0.89ms | Prep: 0.60ms | Inf: 9.85ms | Post: 0.09ms | Total: 11.44ms (87.4 FPS)
Decode: 0.90ms | Prep: 0.66ms | Inf: 9.71ms | Post: 0.09ms | Total: 11.35ms (88.1 FPS)
```

# nms=False end2end=True non-exhaustive fp16 MLIR=1 removed tracker and mask handling

```txt
 Decode: 1.20ms | Prep: 0.68ms | Inf: 8.43ms | Post: 0.18ms | Total: 10.52ms (95.1 FPS) | Stale dropped: 0
 Decode: 0.90ms | Prep: 0.66ms | Inf: 8.42ms | Post: 0.18ms | Total: 10.18ms (98.2 FPS) | Stale dropped: 0
 Decode: 0.83ms | Prep: 0.67ms | Inf: 8.49ms | Post: 0.18ms | Total: 10.19ms (98.2 FPS) | Stale dropped: 0
 Decode: 0.84ms | Prep: 0.66ms | Inf: 8.52ms | Post: 0.19ms | Total: 10.23ms (97.7 FPS) | Stale dropped: 0
 Decode: 0.87ms | Prep: 0.67ms | Inf: 8.55ms | Post: 0.19ms | Total: 10.29ms (97.2 FPS) | Stale dropped: 0
 Decode: 0.85ms | Prep: 0.66ms | Inf: 8.54ms | Post: 0.18ms | Total: 10.25ms (97.6 FPS) | Stale dropped: 0
 Decode: 0.84ms | Prep: 0.67ms | Inf: 8.58ms | Post: 0.18ms | Total: 10.29ms (97.2 FPS) | Stale dropped: 0
 Decode: 1.06ms | Prep: 0.72ms | Inf: 8.51ms | Post: 0.19ms | Total: 10.50ms (95.3 FPS) | Stale dropped: 0
 Decode: 1.15ms | Prep: 0.67ms | Inf: 8.46ms | Post: 0.18ms | Total: 10.47ms (95.5 FPS) | Stale dropped: 0
 Decode: 0.88ms | Prep: 0.68ms | Inf: 8.49ms | Post: 0.18ms | Total: 10.24ms (97.7 FPS) | Stale dropped: 0
 Decode: 0.91ms | Prep: 0.67ms | Inf: 8.54ms | Post: 0.18ms | Total: 10.32ms (96.9 FPS) | Stale dropped: 0
 Decode: 0.95ms | Prep: 0.68ms | Inf: 8.50ms | Post: 0.18ms | Total: 10.33ms (96.8 FPS) | Stale dropped: 0
 Decode: 0.89ms | Prep: 0.69ms | Inf: 8.60ms | Post: 0.19ms | Total: 10.38ms (96.3 FPS) | Stale dropped: 0
 Decode: 0.84ms | Prep: 0.68ms | Inf: 8.54ms | Post: 0.18ms | Total: 10.27ms (97.4 FPS) | Stale dropped: 0
 Decode: 0.89ms | Prep: 0.69ms | Inf: 8.52ms | Post: 0.11ms | Total: 10.20ms (98.0 FPS) | Stale dropped: 0
 Decode: 1.06ms | Prep: 0.68ms | Inf: 8.47ms | Post: 0.09ms | Total: 10.31ms (97.0 FPS) | Stale dropped: 0
```
