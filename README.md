# 🤖 AutoQA — Browser Testing Agent

Write browser tests in plain English. AI drives a real browser, takes screenshots, and verifies with Gemini Vision.

## 🎯 What It Does

Write a test like: Go to Wikipedia, search for AI, verify the article loads
AutoQA plans the steps, drives a real Chromium browser, screenshots every action, and verifies with Gemini Vision.

## 🏗️ Architecture

Plain English -> Llama 3 plans steps -> Playwright drives browser -> Screenshots -> Gemini Vision verifies -> Report

## 🛡️ Safety

- Only public HTTP/HTTPS URLs allowed
- localhost and private IPs blocked
- Max 30 actions per test

## 🚀 Quick Start

pip install -r requirements.txt
playwright install chromium
export GROQ_API_KEY=gsk_your_key
streamlit run app.py

## 🔑 API Keys

- Groq (required) - console.groq.com
- Gemini (optional) - ai.google.dev

## 🧪 Tests

pytest tests/ -v

---
Built with LangGraph + Playwright + Gemini Vision + Streamlit
