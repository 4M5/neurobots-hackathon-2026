# Exam Behavior Anomaly Detection

An anomaly detection prototype for identifying potentially suspicious
behavior during online exams.

Developed as a team project for the IIT Palakkad Hackathon.

## Overview

Online exams generate behavioral signals such as typing patterns, tab
switching, paste events, focus changes, and idle time. This project uses
machine learning to identify behavior that deviates from normal patterns.

The system uses Isolation Forest for anomaly detection and integrates
the Gemini API to generate natural-language explanations of detected
anomalies.

This is a hackathon prototype and is not intended to definitively
determine whether a student is cheating.

## Features

- Behavioral anomaly detection using Isolation Forest
- Analysis of 12 behavioral features
- Synthetic dataset generation
- Model training and testing
- Flask-based web application
- Gemini API integration for anomaly explanations
- Web interface for entering behavioral data and viewing results

## Behavioral Features

The model uses the following behavioral signals:

- Average key interval
- Standard deviation of key intervals
- Typing speed
- Backspace rate
- Paste count
- Tab switches
- Focus loss
- Average idle time
- Maximum idle time
- Answer edits
- Time to first key
- Answer duration

## Machine Learning Approach

The project uses **Isolation Forest**, an unsupervised anomaly detection
algorithm.

The model learns patterns from normal behavioral samples and assigns an
anomaly score to new observations. Samples that significantly differ from
the learned normal behavior are flagged as potential anomalies.

An anomaly does not necessarily indicate cheating. Unusual behavior can
also occur because of legitimate differences between users.

## Dataset

The current implementation uses synthetic behavioral data for
demonstration and experimentation.

- 800 normal samples
- 200 anomalous samples
- 12 behavioral features

Since the dataset is synthetic, the results should not be interpreted as
real-world cheating detection performance.

## Gemini Integration

Gemini is used to generate a natural-language explanation for detected
anomalies.

The API key can be configured through an environment variable:

```text
GEMINI_API_KEY=your_api_key

## Project Structure

```text
.
├── generate_dataset.py
├── train_model.py
├── test_model.py
├── server.py
├── index.html
├── app.js
├── styles.css
├── requirements.txt
└── README.md

## Installation

### 1. Clone the repository

```bash
git clone <your-fork-url>
cd neurobots-hackathon-2026
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n exam_hack python=3.10 -y
conda activate exam_hack
```

Or using Python virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset

```bash
python generate_dataset.py
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Test the model

```bash
python test_model.py
```

### 4. Configure Gemini API

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

### 5. Start the application

```bash
python server.py
```

Open the local Flask URL in your browser.

## Output

The application analyzes behavioral features and predicts whether the
submitted behavior is consistent with normal patterns or is potentially
anomalous.

For anomalous results, Gemini generates a natural-language explanation
of the detected behavioral patterns.

## Limitations

- Uses synthetic behavioral data.
- An anomaly does not necessarily indicate cheating.
- Not validated on real examination data.
- May produce false positives and false negatives.
- Intended as a hackathon prototype, not a production system.

## Future Improvements

- Evaluate on real-world behavioral datasets.
- Add real-time event collection.
- Compare multiple anomaly detection algorithms.
- Improve dashboard visualization.
- Integrate with online examination platforms.

## Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Flask
- Google Gemini API
- HTML
- CSS
- JavaScript

## Project Context

Developed collaboratively as a team project during the **IIT Palakkad Hackathon**.
