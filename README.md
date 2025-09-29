# AI Real Estate Agent — India (10 km Amenities)

## Overview

This Streamlit application is an AI-powered Real Estate Assistant designed for properties in India. It combines natural language processing, location intelligence, real estate listings, news analysis, price forecasting, and interactive visualizations to help users explore properties and nearby amenities within a 10 km radius.

---

## Features

- **Named Entity Recognition (NER):** Extracts location, BHK, and property size from user input.
- **Google Geocoding API:** Converts locations into geographic coordinates.
- **Google Places API:** Retrieves amenities within a 10 km radius of the property.
- **SerpAPI:** Fetches real estate listings and related news articles.
- **Gemini Generative LLM:** Analyzes news and supports conversational chat with users.
- **LSTM-based Forecasting:** Predicts property price trends for 2, 5, and 10 years, enhanced by news sentiment.
- **Interactive Folium Map:** Displays property and nearby amenities with dynamic markers.
- **Plotly Visualizations:** Interactive charts for price trends and analytics.
- **Session State Persistence:** Maintains inputs, results, and chat history during the session.
- **Chat History Management:** Add, delete last message, or clear entire chat history.

---

## Prerequisites

- Python 3.8 or higher  
- Internet connectivity

---

## Installation

1. Clone or download the repository.

2. Install required Python packages:

```bash
pip install streamlit pandas numpy folium plotly requests transformers tensorflow keras
```

3. API keys used:
  - GOOGLE_PLACES_API_KEY
  - SERP_API_KEY
  - GEMINI_API_KEY

## How to use:
1. Enter a property query in natural language (e.g., "3 BHK apartment near MG Road Bangalore").
2. The app extracts location, BHK, and size details.
3. It geocodes the location and fetches nearby amenities.
4. Displays real estate listings and relevant news.
5. Provides news analysis and chat support using Gemini LLM.
6. Shows price forecasts for 2, 5, and 10 years based on LSTM modeling.
7. Explore interactive maps and charts.
8. Manage chat history with options to add, delete, or clear messages.
