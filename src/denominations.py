"""
denominations.py — Official Banknote Specifications
===================================================
Contains specifications for the Mahatma Gandhi (New) Series Indian banknotes
active legal tender: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500.

Reference: Reserve Bank of India (RBI) Banknote Specifications
"""

from typing import Dict, Any

# Official Reserve Bank of India (RBI) specifications for active legal tender
DENOMINATION_INFO: Dict[str, Dict[str, Any]] = {
    "10": {
        "denomination": "₹10",
        "name": "Ten Rupees",
        "color": "Chocolate Brown",
        "motif": "Sun Temple, Konark",
        "dimensions": "63 mm x 123 mm",
        "bleed_lines": "None",
        "year_introduced": 2018,
    },
    "20": {
        "denomination": "₹20",
        "name": "Twenty Rupees",
        "color": "Greenish Yellow",
        "motif": "Ellora Caves",
        "dimensions": "63 mm x 129 mm",
        "bleed_lines": "None",
        "year_introduced": 2019,
    },
    "50": {
        "denomination": "₹50",
        "name": "Fifty Rupees",
        "color": "Fluorescent Blue",
        "motif": "Hampi with Chariot",
        "dimensions": "66 mm x 135 mm",
        "bleed_lines": "None",
        "year_introduced": 2017,
    },
    "100": {
        "denomination": "₹100",
        "name": "One Hundred Rupees",
        "color": "Lavender",
        "motif": "Rani ki Vav (The Queen's Stepwell)",
        "dimensions": "66 mm x 142 mm",
        "bleed_lines": "4 angular bleed lines on both margins",
        "year_introduced": 2018,
    },
    "200": {
        "denomination": "₹200",
        "name": "Two Hundred Rupees",
        "color": "Bright Yellow",
        "motif": "Sanchi Stupa",
        "dimensions": "66 mm x 146 mm",
        "bleed_lines": "4 angular bleed lines with 2 circles on both margins",
        "year_introduced": 2017,
    },
    "500": {
        "denomination": "₹500",
        "name": "Five Hundred Rupees",
        "color": "Stone Grey",
        "motif": "Red Fort with Indian flag",
        "dimensions": "66 mm x 150 mm",
        "bleed_lines": "5 angular bleed lines on both margins",
        "year_introduced": 2016,
    },
}

# Supported class labels
SUPPORTED_CLASSES = ["10", "20", "50", "100", "200", "500"]
