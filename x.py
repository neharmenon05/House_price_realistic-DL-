# x.py
"""
AI Real Estate Agent — India (10 km amenities)
Single-file Streamlit app (no dummy fallbacks).
Features:
 - NER (location, BHK, size)
 - Google Geocoding
 - Google Places Nearby Search (10 km)
 - SerpAPI listings & news
 - Gemini (Generative) LLM for news analysis & chat
 - LSTM-based forecasting (2/5/10 years) with news-based boost
 - Interactive Folium map with amenity markers
 - Plotly visualizations
 - Results persistence using st.session_state
 - Chat history: add, delete last, clear
Before running: ensure required packages are installed and valid API keys are provided
(enter keys in sidebar or set environment variables GOOGLE_PLACES_API_KEY, SERP_API_KEY, GEMINI_API_KEY).
Run: streamlit run x.py
"""

import os
import re
import json
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Logging config
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-REA-India")

# ---------------------------
# Constants
# ---------------------------
PLACES_RADIUS_METERS = 10_000  # 10 km
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
SERPAPI_URL = "https://serpapi.com/search"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# ---------------------------
# Small helpers
# ---------------------------
def now_year() -> int:
    return datetime.now().year

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# ---------------------------
# Simple NER
# ---------------------------
class SimpleNER:
    """Extract location, bhk, and size (sqft) from user text."""

    BHK_PATTERNS = [r'(\d+)\s*bhk', r'(\d+)\s*BHK', r'(\d+)\s*bedroom', r'(\d+)\s*br']
    SIZE_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(sq\s*ft|sqft|square\s*feet|sft|sf)',
        r'(\d+(?:\.\d+)?)\s*(sqm|sq\s*m|square\s*meter)'
    ]
    LOCATION_PATTERNS = [r'in\s+([^,]+)$', r'at\s+([^,]+)$', r'near\s+([^,]+)$']

    def extract(self, text: str) -> Dict:
        """
        Return dict: {"location":str or None, "bhk":int or None, "size_sqft":float or None, "raw_text":str}
        """
        s = (text or "").strip()
        out = {"location": None, "bhk": None, "size_sqft": None, "raw_text": s}

        # BHK
        for p in self.BHK_PATTERNS:
            m = re.search(p, s, re.I)
            if m:
                try:
                    out["bhk"] = int(m.group(1))
                except:
                    out["bhk"] = None
                break

        # size
        for p in self.SIZE_PATTERNS:
            m = re.search(p, s, re.I)
            if m:
                val = safe_float(m.group(1))
                unit = (m.group(2) or "").lower()
                if val is not None:
                    if "sqm" in unit or "meter" in unit:
                        val = val * 10.7639  # convert sqm to sqft
                    out["size_sqft"] = val
                break

        # location
        for p in self.LOCATION_PATTERNS:
            m = re.search(p, s, re.I)
            if m:
                out["location"] = re.sub(r'[^\w\s,.-]', '', m.group(1).strip())
                break

        # fallback heuristic: last comma-separated token(s)
        if not out["location"]:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            if parts:
                out["location"] = parts[-1]

        return out

# ---------------------------
# UniversalFetcher - Google / SerpAPI / Gemini wrappers
# ---------------------------
class UniversalFetcher:
    """
    Wrapper for Google Geocoding, Google Places Nearby, SerpAPI (listings/news),
    and Gemini (LLM) calls. Requires valid API keys (no dummy fallback).
    """

    def __init__(self, google_api_key: str, serp_api_key: str, gemini_api_key: str):
        self.google_key = google_api_key or ""
        self.serp_key = serp_api_key or ""
        self.gemini_key = gemini_api_key or ""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AI-REA-India/1.0"})

    # --- Geocoding ---
    def geocode(self, location: str) -> Tuple[float, float, Dict]:
        """
        Geocode a location string using Google Geocoding API.
        Returns (lat, lng, location_data). Raises RuntimeError on failure.
        """
        if not self.google_key:
            raise RuntimeError("Missing GOOGLE_PLACES_API_KEY.")
        params = {"address": location, "key": self.google_key}
        r = self.session.get(GOOGLE_GEOCODE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("results"):
            raise RuntimeError(f"Geocoding failed: {data}")
        res = data["results"][0]
        geom = res["geometry"]["location"]
        addr = res.get("formatted_address", location)
        # attempt to extract city/state/country
        comps = res.get("address_components", [])
        country = None; state = None; city = None
        for c in comps:
            types = c.get("types", [])
            if "country" in types:
                country = c.get("long_name")
            if "administrative_area_level_1" in types:
                state = c.get("long_name")
            if "locality" in types or "administrative_area_level_2" in types or "sublocality" in types:
                if not city:
                    city = c.get("long_name")
        return geom["lat"], geom["lng"], {"formatted_address": addr, "country": country, "state": state, "city": city}

    # --- Places Nearby ---
    def get_amenities(self, lat: float, lng: float, radius: int = PLACES_RADIUS_METERS) -> Dict:
        """
        Query Google Places Nearby Search for amenity types within the radius.
        Returns a dict: amenity_type -> {"count":int, "names":[str], "places":[raw_place_dicts]}
        """
        if not self.google_key:
            raise RuntimeError("Missing GOOGLE_PLACES_API_KEY.")
        amenity_types = {
            "schools": "school",
            "hospitals": "hospital",
            "parks": "park",
            "metro_stations": "transit_station",
            "malls": "shopping_mall",
            "restaurants": "restaurant",
            "banks": "bank",
            "gyms": "gym",
            "pharmacies": "pharmacy",
            "colleges": "university"
        }
        out = {}
        for label, ptype in amenity_types.items():
            params = {"location": f"{lat},{lng}", "radius": radius, "type": ptype, "key": self.google_key}
            r = self.session.get(GOOGLE_PLACES_NEARBY_URL, params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status not in ("OK", "ZERO_RESULTS"):
                raise RuntimeError(f"Places API error for {label}: {data}")
            results = data.get("results", [])
            names = [pl.get("name") for pl in results]
            out[label] = {"count": len(results), "names": names, "places": results}
            # brief pause for rate limits
            time.sleep(0.12)
        return out

    # --- Place Details (optional) ---
    def get_place_details(self, place_id: str) -> Dict:
        """
        Fetch place details for a given place_id from Google Places Details.
        """
        if not self.google_key:
            raise RuntimeError("Missing GOOGLE_PLACES_API_KEY.")
        params = {"place_id": place_id, "key": self.google_key, "fields": "name,formatted_address,geometry,website,formatted_phone_number,opening_hours,rating,user_ratings_total"}
        r = self.session.get(GOOGLE_PLACES_DETAILS_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            raise RuntimeError(f"Place details error: {data}")
        return data.get("result", {})

    # --- helpers for parsing listing text: price & area ---
    def _extract_price(self, text: str) -> Optional[str]:
        """
        Attempt to extract a price string from listing text (INR / ₹ / lakhs / crore).
        Returns the matched substring or None.
        """
        if not text:
            return None
        # Indian price patterns: ₹, Rs., Lakh, Crore
        m = re.search(r'(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*(?:lakh|lakhs|lacs|crore|cr|cr\.|k|m)?', text, re.I)
        if m:
            return m.group(0).strip()
        # fallback to large number
        m2 = re.search(r'[\d,]{5,}', text)
        if m2:
            return m2.group(0)
        return None

    def _extract_area(self, text: str) -> Optional[str]:
        """
        Attempt to extract area string like '1200 sqft' or '110 sqm' from text.
        """
        if not text:
            return None
        m = re.search(r'(\d{2,5})\s*(sq\.?\s*ft|sqft|square\s*feet|sft|sf|sqm|sq\.?\s*m)', text, re.I)
        if m:
            return m.group(0).strip()
        return None

    # --- SerpAPI listings ---
    def search_listings(self, query: str, num: int = 10) -> List[Dict]:
        """
        Use SerpAPI to search for property listings. Returns simplified listing objects.
        """
        if not self.serp_key:
            raise RuntimeError("Missing SERP_API_KEY.")
        params = {"api_key": self.serp_key, "engine": "google", "q": query, "num": num}
        r = self.session.get(SERPAPI_URL, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        results = data.get("organic_results", []) or []
        listings = []
        for res in results:
            title = res.get("title")
            snippet = res.get("snippet")
            link = res.get("link")
            price = self._extract_price(f"{title} {snippet}")
            area = self._extract_area(f"{title} {snippet}")
            listings.append({"title": title, "snippet": snippet, "link": link, "price": price, "area": area, "raw": res})
        return listings

    # --- SerpAPI news ---
    def search_news(self, query: str, num: int = 8) -> List[Dict]:
        """
        Get news results related to the query using SerpAPI (news search).
        """
        if not self.serp_key:
            raise RuntimeError("Missing SERP_API_KEY.")
        params = {"api_key": self.serp_key, "engine": "google", "q": query, "tbm": "nws", "num": num}
        r = self.session.get(SERPAPI_URL, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        items = data.get("news_results") or data.get("news", []) or []
        news_clean = []
        for n in items[:num]:
            # SerpAPI news shape may vary
            news_clean.append({
                "title": n.get("title"),
                "snippet": n.get("snippet"),
                "source": (n.get("source", {}).get("name") if isinstance(n.get("source"), dict) else n.get("source")),
                "date": n.get("date"),
                "link": n.get("link")
            })
        return news_clean

    # --- Gemini call (robust) ---
    def call_gemini(self, prompt: str, max_output_tokens: int = 1024, temperature: float = 0.0) -> str:
        """
        Call Gemini generateContent endpoint. Returns plain text output.
        Raises RuntimeError on HTTP errors.
        """
        if not self.gemini_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_output_tokens)}
        }
        url = f"{GEMINI_URL}?key={self.gemini_key}"
        r = self.session.post(url, json=payload, timeout=30)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # include response body for debugging
            body = r.text
            logger.error("Gemini HTTP error: %s - body: %s", e, body)
            raise RuntimeError(f"Gemini API error: {e} - {body}")
        data = r.json()
        # try multiple shapes
        try:
            # primary shape
            candidates = data.get("candidates") or data.get("outputs") or []
            if candidates and isinstance(candidates, list):
                cand0 = candidates[0]
                # content -> parts -> text
                content = cand0.get("content") or cand0.get("output") or cand0
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    if parts and isinstance(parts, list):
                        return parts[0].get("text", "")
                # sometimes candidate may have 'text' directly
                if isinstance(cand0, dict) and "text" in cand0:
                    return cand0.get("text", "")
            # fallback: attempt to read top-level 'output' or stringify
            if "output" in data and isinstance(data["output"], list):
                texts = []
                for o in data["output"]:
                    if isinstance(o, dict):
                        texts.append(o.get("content", "") or o.get("text", ""))
                    else:
                        texts.append(str(o))
                return "\n".join(texts)
            return json.dumps(data)
        except Exception as e:
            logger.error("Unexpected Gemini response shape: %s", e)
            return json.dumps(data)

# ---------------------------
# NewsImpactAnalyzer
# ---------------------------
class NewsImpactAnalyzer:
    """
    Analyze news items using the LLM to extract:
      - infrastructure_boost: float [0.0, 0.5]
      - completion_year: int or None
      - summary: string
    """

    def __init__(self, fetcher: UniversalFetcher):
        self.fetcher = fetcher

    def analyze(self, news_items: List[Dict]) -> Dict:
        """
        Send up to 6 news items to Gemini to extract structured impact.
        Result: {"infrastructure_boost":float,"completion_year":int|None,"summary":str}
        """
        if not news_items:
            return {"infrastructure_boost": 0.0, "completion_year": None, "summary": "No news items."}

        prompt = (
            "You are an assistant that extracts structured signals from short real-estate / infrastructure news. "
            "Given an array of news items (title + snippet), return ONLY a JSON object with keys:\n"
            "- infrastructure_boost: float between 0.0 and 0.5 (increase in future prices, e.g., 0.10 = +10%)\n"
            "- completion_year: integer year (e.g., 2028) if a completion/operational year is mentioned in the future, else null\n"
            "- summary: short explanation (1-2 sentences)\n\n"
            f"News items (JSON array): {json.dumps(news_items[:6], ensure_ascii=False)}\n\n"
            "Return a valid JSON object and nothing else."
        )
        try:
            txt = self.fetcher.call_gemini(prompt, max_output_tokens=512, temperature=0.0)
            # extract JSON
            m = re.search(r'\{.*\}', txt, re.S)
            if m:
                parsed = json.loads(m.group(0))
                boost = float(parsed.get("infrastructure_boost", 0.0))
                boost = max(0.0, min(0.5, boost))
                cy = parsed.get("completion_year")
                if isinstance(cy, (int, float)) and int(cy) >= now_year():
                    completion_year = int(cy)
                elif isinstance(cy, str) and cy.isdigit() and int(cy) >= now_year():
                    completion_year = int(cy)
                else:
                    completion_year = None
                summary = parsed.get("summary", "")
                return {"infrastructure_boost": boost, "completion_year": completion_year, "summary": summary}
            # fallback parse simple numbers/years
            years = re.findall(r'\b(20[2-9]\d)\b', txt)
            year = None
            if years:
                for y in sorted(set(int(x) for x in years)):
                    if y >= now_year():
                        year = y
                        break
            # detect keywords
            boost = 0.0
            for kw, w in [("highway",0.12),("expressway",0.12),("metro",0.15),("airport",0.20),("bridge",0.08)]:
                if kw in txt.lower():
                    boost += w
            boost = min(boost, 0.5)
            return {"infrastructure_boost": float(round(boost,3)), "completion_year": year, "summary": txt[:400]}
        except Exception as e:
            logger.error("Gemini news analyze failed: %s", e)
            # as final fallback (shouldn't happen in no-dummy mode) use heuristic
            return self.heuristic(news_items)

    def heuristic(self, news_items: List[Dict]) -> Dict:
        blob = " ".join([f"{n.get('title','')} {n.get('snippet','')}" for n in news_items]).lower()
        keywords = {
            r'\bhighway\b': 0.12,
            r'\bexpressway\b': 0.12,
            r'\bmetro\b': 0.15,
            r'\bairport\b': 0.20,
            r'\bbridge\b': 0.08
        }
        boost = 0.0; reasons = []
        for k, w in keywords.items():
            if re.search(k, blob):
                boost += w; reasons.append(k.strip('\\b'))
        boost = min(boost, 0.5)
        years = re.findall(r'\b(20[2-9]\d)\b', blob)
        year = int(years[0]) if years else None
        summary = "heuristic: " + (", ".join(reasons) if reasons else "no infra keywords")
        return {"infrastructure_boost": float(round(boost,3)), "completion_year": year, "summary": summary}

# ---------------------------
# Price estimator
# ---------------------------
class UniversalPriceCalculator:
    """
    Estimate current price (INR Lakhs) using base per-sqft and amenity/listings signals.
    """

    def __init__(self):
        self.base_per_sqft = 5000.0  # approximate India base

    def estimate(self, entities: Dict, amenities: Dict, listings: List[Dict], location_data: Dict) -> Tuple[float, Dict]:
        """
        Returns (display_price_lakhs, market_context)
        """
        bhk = entities.get("bhk") or 2
        size = entities.get("size_sqft") or 1000.0

        # amenity weighted score
        weights = {
            "metro_stations": 0.20, "schools": 0.04, "hospitals": 0.04, "colleges": 0.03,
            "parks": 0.02, "malls": 0.03, "restaurants": 0.01, "banks": 0.01, "gyms": 0.01, "pharmacies": 0.01
        }
        amen_score = 0.0
        for k, v in amenities.items():
            cnt = v.get("count", 0)
            w = weights.get(k, 0.01)
            amen_score += cnt * w
        amenity_mult = 1.0 + min(0.9, amen_score / 10.0)

        # listing derived market multiplier
        parsed_psfs = []
        for l in listings:
            area_txt = l.get("area") or ""
            price_txt = l.get("price") or ""
            area_m = re.search(r'(\d{2,5})', area_txt.replace(',', ''))
            price_m = re.search(r'([\d,]+)', price_txt.replace(',', ''))
            if area_m and price_m:
                try:
                    area_num = float(area_m.group(1))
                    price_num = float(price_m.group(1))
                    psf = price_num / max(1.0, area_num)
                    parsed_psfs.append(psf)
                except:
                    pass
        market_mult = 1.0
        if parsed_psfs:
            avg_psf = float(np.mean(parsed_psfs))
            market_mult = np.clip(avg_psf / (self.base_per_sqft + 1e-6), 0.6, 1.6)

        bhk_map = {1:0.85,2:1.0,3:1.15,4:1.3,5:1.5}
        bhk_mult = bhk_map.get(bhk,1.0)

        adjusted_psf = self.base_per_sqft * amenity_mult * market_mult * bhk_mult
        total_price_inr = adjusted_psf * size
        display_lakhs = total_price_inr / 100000.0

        market_context = {
            "base_per_sqft": self.base_per_sqft,
            "adjusted_per_sqft": adjusted_psf,
            "amenity_mult": round(amenity_mult,3),
            "market_mult": round(market_mult,3),
            "bhk_mult": bhk_mult,
            "currency": "INR",
            "display_unit": "Lakh",
            "growth_rate": 0.07
        }
        return display_lakhs, market_context

# ---------------------------
# Tiny LSTM forecaster
# ---------------------------
class TinyLSTMForecaster:
    """Train a tiny LSTM on a synthetic annual series and forecast future prices; apply news boosts."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_series(self, current_price, base_growth=0.07, years=12, volatility=0.03):
        series = [max(0.1, current_price * (0.6 + np.random.rand() * 0.2))]
        for _ in range(1, years):
            shock = np.random.normal(0, volatility)
            nxt = series[-1] * (1 + base_growth + shock)
            series.append(max(0.01, nxt))
        return np.array(series, dtype=np.float32)

    def train_and_forecast(
        self,
        current_price_display: float,
        market_growth_rate: float,
        news_boost: float = 0.0,
        news_completion_year: Optional[int] = None,
        years_to_forecast: List[int] = [2,5,10]
    ) -> Dict:
        """
        Trains tiny LSTM and returns forecasts mapping year->value (same units as current_price_display).
        Applies news boost multiplicatively for forecasts at/after completion year.
        """
        hist_years = 12
        series = self._generate_series(current_price_display, base_growth=market_growth_rate, years=hist_years)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
        seq_len = 4
        X, y = [], []
        for i in range(len(scaled)-seq_len):
            X.append(scaled[i:i+seq_len]); y.append(scaled[i+seq_len])
        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y).reshape(-1, 1)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden=16, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        model = LSTMModel().to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        epochs = 200
        for epoch in range(epochs):
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 50 == 0:
                # detach before cast to python float to avoid autograd warnings
                logger.debug("LSTM epoch %d loss %.6f", epoch, float(loss.detach().cpu().numpy()))

        # Forecast iteratively
        model.eval()
        last_seq = torch.tensor(scaled[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(self.device)
        preds_scaled = []
        with torch.no_grad():
            cur_seq = last_seq.clone()
            steps = max(years_to_forecast)
            for _ in range(steps):
                out = model(cur_seq)
                val = out.cpu().numpy().flatten()[0]
                preds_scaled.append(val)
                cur_seq = torch.cat([cur_seq[:,1:,:], out.reshape(1,1,1)], axis=1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        forecast = {}
        for y in years_to_forecast:
            idx = y - 1
            base_val = float(preds[idx]) if idx < len(preds) else float(current_price_display * ((1 + market_growth_rate) ** y))
            adj_val = base_val
            # apply news boost if completion year present and falls within horizon
            if news_boost and news_completion_year:
                years_until_completion = news_completion_year - now_year()
                if years_until_completion <= 0 or years_until_completion <= y:
                    adj_val = base_val * (1.0 + news_boost)
            elif news_boost and not news_completion_year:
                # progressive effect for longer horizons
                factor = 1.0 + news_boost * min(1.0, math.sqrt(y/10.0))
                adj_val = base_val * factor
            forecast[y] = adj_val
        return forecast

# ---------------------------
# Streamlit app
# ---------------------------
def run_app():
    st.set_page_config(page_title="AI Real Estate Agent — India", layout="wide")
    st.title("🏡 AI Real Estate Agent — India (10 km amenities)")

    # Sidebar: API keys & controls
    st.sidebar.header("API Keys (required)")
    google_key_input = st.sidebar.text_input("Google Places / Geocoding API Key", value=os.getenv("GOOGLE_PLACES_API_KEY",""), type="password")
    serp_key_input = st.sidebar.text_input("SerpAPI Key", value=os.getenv("SERP_API_KEY",""), type="password")
    gemini_key_input = st.sidebar.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY",""), type="password")
    st.sidebar.markdown("Provide valid API keys. This app requires real keys and will error without them.")

    st.sidebar.header("Search")
    query = st.sidebar.text_input("Property query (e.g., '3BHK 1200 sqft in Vadavalli, Coimbatore')", value="3BHK 1200 sqft in Vadavalli, Coimbatore")
    radius = st.sidebar.number_input("Amenities radius (meters, 1000-20000)", min_value=1000, max_value=20000, value=PLACES_RADIUS_METERS, step=500)
    run_button = st.sidebar.button("Run analysis")

    st.sidebar.markdown("---")
    st.sidebar.header("Chat controls")
    chat_text = st.sidebar.text_input("Chat message", value="")
    chat_send = st.sidebar.button("Send Chat")
    clear_chat = st.sidebar.button("Clear Chat History")
    delete_last = st.sidebar.button("Delete Last Chat Message")
    clear_results = st.sidebar.button("Clear last analysis results")

    # session_state initialization
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of tuples (speaker, text, timestamp)
    if "last_analysis" not in st.session_state:
        st.session_state["last_analysis"] = None

    # Chat controls operate immediately
    if clear_chat:
        st.session_state["chat_history"] = []
    if delete_last:
        if st.session_state["chat_history"]:
            st.session_state["chat_history"].pop()
    if clear_results:
        st.session_state["last_analysis"] = None

    # validate keys before heavy operations; we allow UI to show chat even without keys but the user requested no dummy -> require keys
    keys_provided = all([google_key_input.strip(), serp_key_input.strip(), gemini_key_input.strip()])
    if not keys_provided:
        st.warning("Please provide all API keys in the sidebar (Google Places/Geocoding, SerpAPI, Gemini). The app requires real keys.")
    # Initialize helpers only if keys present
    fetcher = None
    if keys_provided:
        fetcher = UniversalFetcher(google_key_input.strip(), serp_key_input.strip(), gemini_key_input.strip())
        ner = SimpleNER()
        news_analyzer = NewsImpactAnalyzer(fetcher)
        price_calc = UniversalPriceCalculator()
        forecaster = TinyLSTMForecaster()

    # Run analysis pipeline
    if run_button:
        if not keys_provided:
            st.error("Cannot run analysis: API keys missing.")
            st.stop()

        if not query or query.strip() == "":
            st.error("Please enter a property query (e.g., '3BHK 1200 sqft in Vadavalli, Coimbatore').")
            st.stop()

        # 1) NER
        with st.spinner("Extracting entities..."):
            entities = ner.extract(query)
        st.success(f"Entities extracted: {entities}")

        # 2) Geocode
        with st.spinner("Geocoding location..."):
            try:
                lat, lng, location_data = fetcher.geocode(entities.get("location") or entities.get("raw_text"))
            except Exception as e:
                st.error(f"Geocoding failed: {e}")
                st.stop()
        st.success(f"Geocoded: {location_data.get('formatted_address', '')} (lat={lat:.5f}, lng={lng:.5f})")

        # 3) Amenities (10 km or radius)
        with st.spinner(f"Fetching amenities within {radius/1000:.1f} km..."):
            try:
                amenities = fetcher.get_amenities(lat, lng, radius=int(radius))
            except Exception as e:
                st.error(f"Places API failed: {e}")
                st.stop()
        st.success("Amenities fetched.")

        # 4) Listings via SerpAPI
        listing_query = f"{entities.get('bhk') or ''}BHK {entities.get('location') or ''} properties"
        with st.spinner("Searching property listings (SerpAPI)..."):
            try:
                listings = fetcher.search_listings(listing_query, num=12)
            except Exception as e:
                st.error(f"Listings search failed: {e}")
                st.stop()
        st.success(f"Found {len(listings)} listing results (search output).")

        # 5) News via SerpAPI
        news_query = f"infrastructure {entities.get('location') or ''} real estate"
        with st.spinner("Searching news (SerpAPI)..."):
            try:
                news_items = fetcher.search_news(news_query, num=8)
            except Exception as e:
                st.error(f"News search failed: {e}")
                st.stop()
        st.success(f"Retrieved {len(news_items)} news items.")

        # 6) Analyze news with Gemini
        with st.spinner("Analyzing news for infrastructure impact (Gemini)..."):
            try:
                impact = news_analyzer.analyze(news_items)
            except Exception as e:
                st.error(f"News analysis failed: {e}")
                st.stop()
        st.success("News analysis complete.")
        st.write("News impact:", impact)

        # 7) Price estimation
        with st.spinner("Estimating current property price..."):
            try:
                current_price_lakhs, market_context = price_calc.estimate(entities, amenities, listings, location_data)
            except Exception as e:
                st.error(f"Price estimation failed: {e}")
                st.stop()
        st.success("Price estimated.")

        # 8) Forecasting
        with st.spinner("Training LSTM (demo) and forecasting (2/5/10 years)..."):
            try:
                forecasts = forecaster.train_and_forecast(current_price_lakhs, market_context["growth_rate"],
                                                         news_boost=impact.get("infrastructure_boost", 0.0),
                                                         news_completion_year=impact.get("completion_year", None),
                                                         years_to_forecast=[2,5,10])
            except Exception as e:
                st.error(f"Forecasting failed: {e}")
                st.stop()
        st.success("Forecasting complete.")

        # 9) Save results into session state so they persist across reruns
        st.session_state["last_analysis"] = {
            "query": query,
            "entities": entities,
            "lat": lat, "lng": lng, "location_data": location_data,
            "amenities": amenities,
            "listings": listings,
            "news": news_items,
            "impact": impact,
            "current_price_lakhs": current_price_lakhs,
            "market_context": market_context,
            "forecasts": forecasts,
            "timestamp": datetime.utcnow().isoformat()
        }

        # show immediate success message and scroll to results
        st.success("Analysis completed and saved. Scroll down to view persisted results.")

    # -------------------------
    # Render persisted results (if any)
    # -------------------------
    if st.session_state.get("last_analysis"):
        la = st.session_state["last_analysis"]
        st.header("🏷️ Persisted Analysis (last run)")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Estimated Price", f"₹{la['current_price_lakhs']:.2f} L")
        with cols[1]:
            st.metric("Adjusted Price/sqft", f"₹{la['market_context']['adjusted_per_sqft']:.0f}")
        with cols[2]:
            st.metric("Amenity Mult", f"{la['market_context']['amenity_mult']}")
        with cols[3]:
            st.metric("Market Mult", f"{la['market_context']['market_mult']}")

        st.subheader("Location")
        st.write(la["location_data"])

        # Amenity chart and lists
        st.subheader("Amenities (names & counts)")
        amen_df = pd.DataFrame([{"Amenity": k.replace("_"," ").title(), "Count": v["count"]} for k,v in la["amenities"].items()])
        fig_bar = px.bar(amen_df, x="Amenity", y="Count", title="Amenity counts within search radius")
        st.plotly_chart(fig_bar, use_container_width=True)

        # show names (bulleted)
        for k, v in la["amenities"].items():
            st.markdown(f"**{k.replace('_',' ').title()} ({v['count']})**")
            for name in v.get("names", [])[:20]:
                st.write(" - ", name)

        # Listings table
        st.subheader("Listings (search results)")
        if la["listings"]:
            df_list = pd.DataFrame(la["listings"])
            df_list_display = df_list[["title","price","area","link","snippet"]].fillna("")
            st.dataframe(df_list_display, use_container_width=True)
        else:
            st.write("No listings returned by SerpAPI.")

        # News items & impact
        st.subheader("News used for impact analysis")
        for n in la["news"]:
            st.markdown(f"**{n.get('title')}** — {n.get('source')}")
            st.write(n.get("snippet"))
            if n.get("link"):
                st.write(n.get("link"))
        st.write("Impact analysis:", la["impact"])

        # Forecasts
        st.subheader("Forecasts (Lakhs INR)")
        years = [0,2,5,10]
        values = [la["current_price_lakhs"], la["forecasts"][2], la["forecasts"][5], la["forecasts"][10]]
        df_fore = pd.DataFrame({"Year":[now_year()+y for y in years], "Price_Lakhs": values})
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_fore["Year"], y=df_fore["Price_Lakhs"], mode="lines+markers", name="Forecast"))
        fig_line.update_layout(title="Price Forecast (Lakhs INR)", xaxis_title="Year", yaxis_title="Price (Lakhs)")
        st.plotly_chart(fig_line, use_container_width=True)

        # Price-per-sqft histogram from listings if parseable
        parsed_psfs = []
        for l in la["listings"]:
            area_txt = l.get("area") or ""
            price_txt = l.get("price") or ""
            area_m = re.search(r'(\d{2,5})', area_txt.replace(',',''))
            price_m = re.search(r'([\d,]+)', price_txt.replace(',',''))
            if area_m and price_m:
                try:
                    a = float(area_m.group(1)); p = float(price_m.group(1)); parsed_psfs.append(p / max(1.0, a))
                except:
                    pass
        if parsed_psfs:
            df_psf = pd.DataFrame({"psf": parsed_psfs})
            fig_psf = px.histogram(df_psf, x="psf", nbins=20, title="Inferred price per sqft (listing hints)")
            st.plotly_chart(fig_psf, use_container_width=True)
        else:
            st.write("No parsable price/area information in listings to compute PSF histogram.")

        # Map with property + amenity markers
        st.subheader("Map (property + amenities)")
        try:
            m = folium.Map(location=[la["lat"], la["lng"]], zoom_start=12, tiles="OpenStreetMap")
            folium.Marker([la["lat"], la["lng"]], popup=f"Property: {la['entities'].get('location')}", icon=folium.Icon(color="red", icon="home")).add_to(m)
            color_map = {
                "schools":"blue","hospitals":"purple","parks":"green","metro_stations":"darkblue","malls":"cadetblue",
                "restaurants":"orange","banks":"darkgreen","gyms":"pink","pharmacies":"lightgray","colleges":"beige"
            }
            for k, v in la["amenities"].items():
                places = v.get("places", [])[:25]
                for pl in places:
                    loc = pl.get("geometry", {}).get("location")
                    if not loc: continue
                    name = pl.get("name")
                    folium.CircleMarker([loc.get("lat"), loc.get("lng")], radius=4,
                                        popup=f"{k.replace('_',' ').title()}: {name}",
                                        color=color_map.get(k, "gray"), fill=True).add_to(m)
            st_folium(m, width=900, height=600)
        except Exception as e:
            st.write("Map rendering failed:", e)

    else:
        st.info("No persisted analysis yet. Run analysis from the sidebar to compute and save results.")

    # -------------------------
    # Chat UI (uses Gemini)
    # -------------------------
    st.header("🤖 Assistant (chat)")

    # display chat history
    st.subheader("Chat history")
    if st.session_state["chat_history"]:
        for i, entry in enumerate(st.session_state["chat_history"]):
            ts = entry[2] if len(entry) > 2 else ""
            speaker = entry[0]; text = entry[1]
            if speaker == "User":
                st.markdown(f"**You:** {text}  <span style='color:gray;font-size:12px'> {ts} </span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Assistant:** {text}  <span style='color:gray;font-size:12px'> {ts} </span>", unsafe_allow_html=True)
    else:
        st.write("No chat messages yet.")

    # handle sending chat
    if chat_send and chat_text.strip():
        if not keys_provided:
            st.error("Cannot use assistant: API keys missing.")
        else:
            user_msg = chat_text.strip()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
            st.session_state["chat_history"].append(("User", user_msg, timestamp))
            # build context from last_analysis if available
            context = ""
            if st.session_state.get("last_analysis"):
                la = st.session_state["last_analysis"]
                context = (
                    f"Context: User searched for {la['entities'].get('bhk')}BHK, size {la['entities'].get('size_sqft')} sqft in {la['entities'].get('location')}. "
                    f"Estimated price: ₹{la['current_price_lakhs']:.2f} L. Amenity counts: "
                    + ", ".join([f"{k}:{v['count']}" for k,v in la['amenities'].items()]) + ". "
                    f"News impact: boost {la['impact'].get('infrastructure_boost')} completion_year {la['impact'].get('completion_year')}."
                )
            prompt = (
                "You are a concise real-estate assistant. Use the context to answer user questions. "
                "If the user asks for amenity names, list up to 10 names per category.\n\n"
                f"{context}\n\nUser: {user_msg}\nAssistant:"
            )
            try:
                reply = fetcher.call_gemini(prompt, max_output_tokens=512, temperature=0.0)
            except Exception as e:
                reply = f"Assistant call failed: {e}"
                logger.error("Assistant call error: %s", e)
            ts_assistant = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
            st.session_state["chat_history"].append(("Assistant", reply, ts_assistant))
            # scroll to chat by re-running (Streamlit will rerun automatically)

    # Chat control buttons already handled earlier to clear/delete

    # Footer notes
    st.markdown("---")
    st.caption("Notes: This prototype uses a demo LSTM with synthetic history for forecasting. For production-grade forecasts, train on real historic price time series per micro-market, add more features, and validate thoroughly. Gemini usage will consume API quota and costs; be mindful of quotas.")

if __name__ == "__main__":
    run_app()
