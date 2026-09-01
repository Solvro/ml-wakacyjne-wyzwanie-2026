#!/usr/bin/env python
import os
from dotenv import load_dotenv

# model key loading
_ = load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# check if loaded successfully
if not api_key:
    raise ValueError("No key found!")
else:
    print("Key loaded!")
