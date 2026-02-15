🐯 TORA – RAG-Based Running Biomechanics Agent

TORA is a Retrieval-Augmented Generation (RAG) chatbot designed to analyze running biomechanics using high-speed video data and STRYD’s footpath framework.

It combines computer vision, biomechanical feature extraction, structured knowledge bases, and a 3-layer reasoning model to generate validated, context-aware responses about running mechanics.

🧠 System Overview

The system connects three domains:

Computer Vision

Biomechanical Representation

Knowledge-Grounded Reasoning (RAG)

High-speed treadmill footage is converted into a structured biomechanical representation.
That representation is aligned with STRYD’s footpath definitions.
A RAG agent retrieves relevant expert knowledge and produces validated answers.

🎥 Data Pipeline
1. Video Acquisition

210 FPS ELP synchronous stereo camera

Rear treadmill view

Stereo frame split into left/right

Preprocessing is handled in:

vid.pp.py


It:

Extracts a centered 5-second segment (1050 frames)

Crops the requested stereo side

Saves a mono clip for pose estimation

2. Pose Extraction

A YOLO Human Pose model (COCO 17 keypoints) extracts the skeleton per frame.

The ankle joint is used to define the footpath trajectory.

3. Joint Angle Computation

For each frame:

Ankle angle

Knee angle

Hip flag (binary classification state)

Example structure:

frame,x,y,ankle_angle,knee_angle,hip_flag,point_type
0,208.53,172.92,138.08,146.15,0,2
...


This file:

footpath_ankle_sequence_with_angles.csv

4. Footpath Characteristic Classification

This is the critical step.

Joint angle patterns are mapped to STRYD’s defined footpath phases:

Toe-Off

Peak Follow Through

Max Heel Recovery

Peak Knee Lift

Foot-Strike

Definitions are derived from STRYD’s official footpath framework.

The extracted representation is condensed into:

footpath_representation.csv


Example:

point_type,avg_x,avg_y
1,195.85,177.69
2,205.44,185.13
3,181.32,199.26
...


This structured representation enables alignment with STRYD’s biomechanical knowledge base.

📚 Knowledge Base Architecture

Two persistent ChromaDB collections are used:

1. run

Contains:

STRYD footpath definitions

Running power concepts

Critical Power framework

Mechanical interpretations

Source: run.txt

2. runner

Contains:

Athlete-specific scientific profile

Tagged biomechanical and physiological information

Built using:

chroma_builder.py

Tag-Aware Summarization

Each <tag>...</tag> block is treated as a semantic unit.

The tag acts as:

A structural summary

A retrieval anchor

A domain-specific index

Embeddings generated with:

nomic-embed-text


Vector similarity:

cosine distance

🤖 RAG Agent Architecture

Implemented in:

agent.py
context_builder.py

Retrieval

For a given query:

Embed user question

Retrieve top-k sections from:

run

runner

Construct structured context

🧠 Three-Layer Reasoning Model

The agent performs internal staged reasoning:

1. Reason

Analyze retrieved evidence

Generate structured implications

No direct answer generation

2. Evaluate

Validate logical consistency

Identify missing data

Check relevance

3. Draw Conclusion

Use validated reasoning only

Generate final answer

Do not expose internal reasoning

This reduces hallucination and enforces evidence-based responses.

🖥 Interface

Built with:

Streamlit

Chat-style interface

Expandable “Agent Thinking” panel for transparency

Run with:

streamlit run agent.py

📊 Experimental Notebooks
footpath_testing.ipynb

Experiments on ankle trajectory

Strategy validation for footpath characteristic extraction

footpath.ipynb

Visualization of classified footpath characteristics

Spatial inspection of extracted key points

🏗 Tech Stack

Python

OpenCV

YOLO Human Pose (COCO 17)

ChromaDB (persistent vector DB)

Ollama

nomic-embed-text

Llama 3.2

Streamlit

Pandas

🔬 Conceptual Foundations

The system integrates:

STRYD Footpath biomechanics model

Running Power theory

Critical Power framework

Retrieval-Augmented Generation

Structured multi-stage reasoning

It bridges measured motion and domain knowledge.

🎯 Purpose

TORA is designed to:

Analyze running mechanics

Interpret footpath characteristics

Connect biomechanics with STRYD metrics

Provide structured, validated feedback

Reduce hallucination in biomechanical interpretation

This is not a generic chatbot.
It is a domain-constrained biomechanical reasoning system.

🚀 Future Improvements

Direct integration of pose inference pipeline

Automated joint-angle classification refinement

Multi-camera 3D reconstruction

STRYD Duo left/right symmetry integration

Quantitative fatigue detection modeling

📌 Summary

TORA converts:

High-speed video → Pose → Joint Angles → Footpath Representation →
Knowledge Retrieval → Structured Reasoning → Validated Insight

It is a biomechanics-first RAG agent built for serious analysis of running form.

The system stands at an interesting intersection: computer vision, vector databases, and human movement science. That intersection is where the interesting things tend to happen.