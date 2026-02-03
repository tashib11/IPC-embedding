# IPC Project — YOLOv5 via Shared Memory (Windows)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

This repository demonstrates a simple producer/consumer object-detection pipeline that shares image frames between a C producer and a Python consumer using shared memory. The producer runs YOLOv5 inference (ONNX) and writes frames; the consumer reads and displays them.

**Contents**

- `producer_windows.c` — C/C++ producer: capture -> inference -> write shared memory
- `consumer_windows.py` — Python consumer: read shared memory -> display
- `yolov5s.onnx` — ONNX model used for inference

**Table of Contents**

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start (Windows)](#quick-start-windows)
- [Build & Run Details](#build--run-details)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Overview

The system uses a flag byte plus raw image bytes in shared memory.

- Producer (`producer_windows.c`): captures frames (file or webcam), runs ONNX-based YOLOv5 inference, draws boxes, and writes the image buffer and a flag byte to shared memory.
- Consumer (`consumer_windows.py`): waits for the flag, reads image bytes, displays them, and clears the flag.

## Architecture

```mermaid
flowchart TD
    subgraph PRODUCER ["C PRODUCER"]
        P1[Capture Image/Webcam] --> P2[YOLOv5 (ONNX) Inference]
        P2 --> P3[Draw Bounding Boxes]
        P3 --> P4{Flag == 0}
        P4 -->|Yes| P5[Write Frame to Shared Memory]
        P5 --> P6[Set Flag = 1]
        P6 --> P1
    end

    subgraph MEMORY ["SHARED MEMORY"]
        M1[Flag Byte]
        M2[Image Data]
    end

    subgraph CONSUMER ["PYTHON CONSUMER"]
        C1{Flag == 1} -->|Yes| C2[Read Frame]
        C2 --> C3[Display Window]
        C3 --> C4[Set Flag = 0]
        C4 --> C1
    end

    P5 --> M2
    P6 --> M1
    M1 -.-> C1
    M2 -.-> C2
    C4 -.-> M1
```

## Quick Start (Windows)

1. Ensure you have a C++ toolchain (MSVC or MinGW) and Python 3.8+ installed.
2. Install Python deps:

```powershell
python -m pip install --upgrade pip
pip install opencv-python numpy
```

3. Compile the producer (see options below).
4. Start the producer, then run the consumer:

```powershell
.\producer_windows.exe
python consumer_windows.py
```

## Build & Run Details

Windows (MinGW / g++) example (if OpenCV is configured with pkg-config):

```powershell
g++ -x c++ producer_windows.c -o producer_windows.exe `pkg-config --cflags --libs opencv4`
```

If using Visual Studio / MSVC, create a Visual Studio project and link against OpenCV's include/libs (or compile with `cl` and the appropriate `/I` and `/link` flags). The source uses C-style APIs and simple structs, but OpenCV's linking requires a C++ toolchain.

Run the producer first then the consumer:

```powershell
.\producer_windows.exe
python consumer_windows.py
```

## Dependencies

- System: Windows 10/11 (tested)
- C++: Visual Studio or MinGW with OpenCV (built for your compiler)
- Python packages:

```powershell
pip install opencv-python numpy
```

- Model: Place `yolov5s.onnx` in the repository root (already included).

## Troubleshooting

- "Shared memory not found" — ensure the producer is running and has created the memory region before starting the consumer.
- "OpenCV linking errors" — verify OpenCV was built for your compiler (MSVC vs MinGW) and use matching include/lib paths.
- Image looks corrupted — check width/height/stride encoding between producer and consumer; they must agree on the pixel format (BGR) and dimensions.
- Performance: ONNX inference may be slow without an optimized runtime (consider ONNX Runtime with CUDA/DirectML accelerators).

## Notes

- This is a minimal demonstration; for production use consider proper synchronization primitives, error handling, and a robust IPC protocol.
- If you want, I can add a `requirements.txt`, a `build.bat` for Visual Studio, or an ONNX Runtime example.

---

If you'd like, I can now:

- create `requirements.txt` and a `build.bat` for Windows (MSVC), or
- add a short `CONTRIBUTING.md` with development notes.
