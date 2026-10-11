# Indian Currency Dataset Guide

This directory holds dataset images structured as follows:

```
dataset/
├── denomination/
│   ├── 10/       (₹10 notes)
│   ├── 20/       (₹20 notes)
│   ├── 50/       (₹50 notes)
│   ├── 100/      (₹100 notes)
│   ├── 200/      (₹200 notes)
│   └── 500/      (₹500 notes)
└── authenticity/
    ├── genuine/     (Genuine Indian currency notes)
    └── counterfeit/ (Counterfeit / Forgery samples)
```

## Guidelines for Dataset Collection
- Resolution: Minimum 224x224 pixels.
- Variety: Capture notes under different lighting conditions, angles, and backgrounds.
- Both front and back sides of currency notes are encouraged.
