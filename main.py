"""Passport Photo Bot — single-file Railway deployment entry point.

All bot configuration, persistence, image processing, UI, handlers, and startup
logic live in this file. Sensitive values are read from environment variables.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import logging
import math
import os
import re
import shutil
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

import cv2
import numpy as np
import aiosqlite
import pillow_heif
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (BotCommand, BufferedInputFile, CallbackQuery, InlineKeyboardButton,
                          InlineKeyboardMarkup, InputMediaPhoto, Message, TelegramObject)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

load_dotenv()
pillow_heif.register_heif_opener()

# ============================== Configuration ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
TEMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "bot.db"
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
try:
    ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "8753914631").replace(" ", "").split(",") if x}
except ValueError:
    ADMIN_IDS = set()
MODE = os.getenv("MODE", "polling").lower().strip()
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
TEMP_TTL_MINUTES = max(1, int(os.getenv("TEMP_TTL_MINUTES", "30")))
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en") if os.getenv("DEFAULT_LANG", "en") in {"en", "hi"} else "en"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
WORKING_PX = 900
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
FLOOD_WINDOW_S = 3.0
FLOOD_MAX_EVENTS = 6
LANGS = ("en", "hi")
TEMPLATE_PATH = BASE_DIR / "templates" / "indian_passport_a4.json"

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

config = SimpleNamespace(**{k: v for k, v in {
    "BASE_DIR": BASE_DIR, "DATA_DIR": DATA_DIR, "TEMP_DIR": TEMP_DIR, "DB_PATH": DB_PATH,
    "BOT_TOKEN": BOT_TOKEN, "ADMIN_IDS": ADMIN_IDS, "MODE": MODE, "WEBHOOK_URL": WEBHOOK_URL,
    "WEBHOOK_HOST": WEBHOOK_HOST, "WEBHOOK_PORT": WEBHOOK_PORT,
    "MAX_CONCURRENT_JOBS": MAX_CONCURRENT_JOBS, "TEMP_TTL_MINUTES": TEMP_TTL_MINUTES,
    "DEFAULT_LANG": DEFAULT_LANG, "LOG_LEVEL": LOG_LEVEL, "WORKING_PX": WORKING_PX,
    "MAX_FILE_BYTES": MAX_FILE_BYTES, "MAX_PIXELS": MAX_PIXELS, "FLOOD_WINDOW_S": FLOOD_WINDOW_S,
    "FLOOD_MAX_EVENTS": FLOOD_MAX_EVENTS, "LANGS": LANGS, "TEMPLATE_PATH": TEMPLATE_PATH,
}.items()})
config.ensure_dirs = ensure_dirs

# ============================== Localization ==============================
_LOCALES = {
    'en': {'lang_name': 'English', 'choose_language': 'Choose your language / अपनी भाषा चुनें', 'welcome': 'Welcome to Passport Photo Bot 📷\n\nSend any photo and get a compliant passport photo plus a print-ready A4 sheet in seconds.\n\n3 tips for the best result:\n1. Stand in plain, even light against a plain wall.\n2. Look straight at the camera, no smile, no shadow.\n3. Send the photo as a <b>File / Document</b> (not a compressed photo) for full quality.\n\nNow send a photo to begin.', 'privacy_note': 'Privacy: your photos are processed in memory and temp files, then deleted within {ttl} minutes, on /cancel, or right after export. Nothing is shared. Type /privacy for details or use "Delete my data" in /settings.', 'help': 'Send a photo (as photo or file). I will find the face, remove the background, fit the size and show a preview.\n\nCommands:\n/size — photo size preset\n/background — background colour\n/copies — copies on the sheet\n/paper — paper size\n/language — हिन्दी / English\n/reset — reset all edits\n/cancel — clear current session\n/privacy — what happens to your data', 'privacy_full': 'Your photos are processed only to make your passport photo. They are stored as temp files and deleted within {ttl} minutes, on /cancel, or after export. Settings (language, presets) are stored in a small database and can be wiped any time with "Delete my data" in /settings. No photo content is logged.', 'send_file_hint': 'For full quality, send the photo as a File / Document. Telegram compresses normal photos to about 1280 px.', 'file_too_large': 'That file is too large for a Telegram bot (over 20 MB). Please downsize it or send a smaller copy.', 'bad_file_type': 'I can read JPG, PNG, WEBP and HEIC photos. That file type is not supported.', 'no_face': 'No face found. Stand in even light, face the camera straight on, and send the photo again — as a File for best quality.\n\nOr tap Manual crop to place the photo yourself.', 'multi_face': 'I found {n} faces. Which person should I use? Tap the number.', 'manual_crop': 'Manual crop', 'stage_checking': '⏳ Checking photo…', 'stage_face': '⏳ Finding face…', 'stage_bg': '⏳ Removing background…', 'stage_fit': '⏳ Fitting size…', 'stage_finish': '⏳ Finishing…', 'queue_position': 'You are number {n} in the queue…', 'card_title': '<b>{size_name} · {w} × {h} mm · {dpi} DPI</b>\nBackground: {bg}   Glow: {glow}   Copies: {copies} on {paper}', 'check_face': 'Face found', 'check_head': 'Head size in range', 'check_center': 'Face centred', 'check_tilt': 'Head not tilted', 'check_eyes': 'Eyes open', 'check_bg': 'Background uniform', 'check_bright': 'Brightness OK', 'check_res': 'Resolution sufficient', 'check_shadow': 'No heavy face shadows', 'menu_size': 'Size', 'menu_background': 'Background', 'menu_enhance': 'Enhance', 'menu_crop': 'Crop', 'menu_sheet': 'Sheet', 'menu_download': '⬇️ Download', 'menu_compare': 'Compare', 'menu_guides': 'Guide lines', 'menu_fix_auto': 'Fix automatically', 'back': '« Back', 'reset': 'Reset', 'undo': '↩️ Undo', 'redo': '↪️ Redo', 'size_title': 'Choose the photo size:', 'size_custom_prompt': 'Send the size as width × height, for example <code>35x45 mm</code>, <code>2x2 in</code> or <code>413x531 px</code>.', 'size_bad': 'I could not read that size. Try e.g. <code>35x45 mm</code> or <code>2x2 in</code>.', 'bg_title': 'Choose the background colour:', 'bg_custom_prompt': 'Send a HEX colour code, e.g. <code>#FF5733</code> — or send a photo and I will pick its average colour.', 'bg_bad_hex': 'That is not a valid HEX colour. Example: <code>#4A90D9</code>', 'bg_transparent': 'Transparent (PNG)', 'enhance_title': 'Enhance — tap − / + to adjust. Preview updates after every tap.', 'enh_glow': 'Glow', 'enh_exposure': 'Exposure', 'enh_brightness': 'Brightness', 'enh_contrast': 'Contrast', 'enh_saturation': 'Saturation', 'enh_warmth': 'Warmth', 'enh_sharpness': 'Sharpness', 'enh_smooth': 'Skin smoothing', 'enh_denoise': 'Denoise', 'enh_look': 'Look', 'look_natural': 'Natural', 'look_bright': 'Bright', 'look_studio': 'Studio', 'look_bw': 'Black & white', 'enh_auto': 'Auto enhance', 'enh_boost': 'Quality boost', 'crop_title': 'Crop — move, zoom, rotate. Fixed to {size}.', 'crop_step': 'Step: {step}', 'crop_autofit': 'Auto-fit face', 'sheet_title': 'Sheet — paper, copies and layout.', 'sheet_paper': 'Paper: {paper}', 'sheet_copies': 'Copies: {copies}', 'sheet_maxfit': 'Max fit', 'sheet_margins': 'Margins: {m} mm', 'sheet_gap': 'Gap: {g} mm', 'sheet_cuts_on': 'Cut marks: on', 'sheet_cuts_off': 'Cut marks: off', 'sheet_border_on': 'Border: on', 'sheet_border_off': 'Border: off', 'sheet_type_number': 'Or just type a number of copies.', 'copies_overflow': '{n} copies do not fit on {paper} (max {max}). Try Max fit or a bigger paper.', 'download_done': 'Done ✅\nPrint tip: print at <b>100% / Actual size</b>, no scaling.', 'target_kb_title': 'Target file size for online forms:', 'target_kb_done': 'Saved at {kb} KB.', 'compare_caption': 'Original (left) vs processed (right)', 'session_cleared': 'Session cleared. Send a new photo whenever you are ready.', 'reset_done': 'All edits reset to automatic.', 'session_expired': 'That session expired. Send the photo again to continue.', 'resume_or_new': 'Welcome back. Resume your last session or send a new photo to start fresh.', 'resume': 'Resume', 'retry': 'Retry', 'error_generic': 'Processing failed at the last step. Tap Retry or send the photo again.', 'settings_title': 'Settings', 'set_dpi': 'Export DPI: {dpi}', 'set_delete_data': 'Delete my data', 'set_preset_save': 'Save as my preset', 'set_preset_use': 'Use my preset', 'preset_saved': 'Preset saved.', 'preset_applied': 'Preset applied.', 'no_preset': 'No preset saved yet.', 'data_deleted': 'All your data and files are deleted.', 'admin_only': 'Admins only.', 'admin_stats': 'Users: {users}\nJobs today: {jobs}\nErrors today: {errors}\nAvg processing: {avg}s', 'admin_health': 'OK. Queue: {q}, temp files: {t}', 'broadcast_prompt': 'Send the message to broadcast to all users.', 'broadcast_done': 'Broadcast sent to {n} users.', 'cancel': 'Cancelled.', 'bg_white': 'White', 'bg_offwhite': 'Off-white', 'bg_lightgrey': 'Light grey', 'bg_lightblue': 'Light blue', 'bg_skyblue': 'Sky blue', 'bg_red': 'Red', 'bg_yellow': 'Yellow', 'bg_green': 'Green', 'bg_black': 'Black', 'bg_custom': 'Custom colour…', 'size_in_passport': 'Indian Passport 35×45', 'size_in_visa': 'Indian Visa / OCI 51×51', 'size_pan': 'PAN card 25×35', 'size_stamp': 'Stamp 20×25', 'size_us': 'US passport 2×2 in', 'size_schengen': 'UK / Schengen 35×45', 'size_custom': 'Custom…', 'restore_more': 'Restore more', 'erase_more': 'Erase more', 'feather': 'Edge softness', 'edge_shift': 'Edge in/out', 'bg_edge_title': 'Edge quality — soften, grow or shrink the mask edge.', 'text_strip_prompt': 'Send the text for the strip under the photo (name and date), or tap Back.', 'delete_data_confirm': 'This deletes your settings, presets and session. Tap again to confirm.', 'album_processing': 'Processing {n} photos — one moment…'},
    'hi': {'lang_name': 'हिन्दी', 'choose_language': 'Choose your language / अपनी भाषा चुनें', 'welcome': 'पासपोर्ट फोटो बॉट में स्वागत है 📷\n\nकोई भी फोटो भेजें और कुछ ही सेकंड में मानक पासपोर्ट फोटो और प्रिंट के लिए तैयार A4 शीट पाएँ।\n\nबेहतर नतीजे के 3 तरीके:\n1. सादी दीवार के सामने, समान रोशनी में खड़े हों।\n2. कैमरे की तरफ सीधे देखें, बिना मुस्कान, बिना परछाई।\n3. पूरी क्वालिटी के लिए फोटो को <b>File / Document</b> के रूप में भेजें।\n\nशुरू करने के लिए एक फोटो भेजें।', 'privacy_note': 'निजता: आपकी फोटो सिर्फ मेमोरी और अस्थायी फाइलों में संसाधित होती हैं और {ttl} मिनट के भीतर, /cancel पर, या एक्सपोर्ट के तुरंत बाद मिटा दी जाती हैं। विवरण के लिए /privacy देखें।', 'help': 'फोटो भेजें (फोटो या फाइल के रूप में)। मैं चेहरा पहचानकर बैकग्राउंड हटाकर साइज़ फिट करूँगा और प्रीव्यू दिखाऊँगा।\n\nकमांड:\n/size — फोटो साइज़\n/background — बैकग्राउंड रंग\n/copies — शीट पर कॉपी\n/paper — पेपर साइज़\n/language — हिन्दी / English\n/reset — सारे बदलाव रीसेट\n/cancel — सत्र साफ़ करें\n/privacy — आपके डेटा की जानकारी', 'privacy_full': 'आपकी फोटो सिर्फ पासपोर्ट फोटो बनाने के लिए संसाधित होती है। ये अस्थायी फाइलों में रहती हैं और {ttl} मिनट के भीतर, /cancel पर या एक्सपोर्ट के बाद मिट जाती हैं। सेटिंग्स छोटे डेटाबेस में रहती हैं और /settings में "मेरा डेटा मिटाएँ" से कभी भी हटाई जा सकती हैं। फोटो की सामग्री लॉग नहीं होती।', 'send_file_hint': 'पूरी क्वालिटी के लिए फोटो को File / Document के रूप में भेजें। Telegram सामान्य फोटो को लगभग 1280 px तक सिकोड़ देता है।', 'file_too_large': 'यह फाइल Telegram बॉट के लिए बहुत बड़ी है (20 MB से अधिक)। कृपया छोटी कॉपी भेजें।', 'bad_file_type': 'मैं JPG, PNG, WEBP और HEIC फोटो पढ़ सकता हूँ। यह फाइल प्रकार समर्थित नहीं है।', 'no_face': 'कोई चेहरा नहीं मिला। समान रोशनी में खड़े हों, कैमरे की ओर सीधे देखें और फोटो दोबारा भेजें — बेहतर क्वालिटी के लिए File के रूप में।\n\nया खुद क्रॉप करने के लिए मैन्युअल क्रॉप दबाएँ।', 'multi_face': 'मुझे {n} चेहरे मिले। किस व्यक्ति का उपयोग करूँ? नंबर दबाएँ।', 'manual_crop': 'मैन्युअल क्रॉप', 'stage_checking': '⏳ फोटो जाँची जा रही है…', 'stage_face': '⏳ चेहरा खोजा जा रहा है…', 'stage_bg': '⏳ बैकग्राउंड हटाया जा रहा है…', 'stage_fit': '⏳ साइज़ फिट किया जा रहा है…', 'stage_finish': '⏳ अंतिम चरण…', 'queue_position': 'आप कतार में नंबर {n} पर हैं…', 'card_title': '<b>{size_name} · {w} × {h} mm · {dpi} DPI</b>\nबैकग्राउंड: {bg}   ग्लो: {glow}   कॉपी: {copies} ({paper})', 'check_face': 'चेहरा मिला', 'check_head': 'सिर का आकार सही', 'check_center': 'चेहरा केंद्र में', 'check_tilt': 'सिर सीधा', 'check_eyes': 'आँखें खुली', 'check_bg': 'बैकग्राउंड एक समान', 'check_bright': 'रोशनी ठीक', 'check_res': 'रिज़ॉल्यूशन पर्याप्त', 'check_shadow': 'चेहरे पर गहरी परछाई नहीं', 'menu_size': 'साइज़', 'menu_background': 'बैकग्राउंड', 'menu_enhance': 'सुधार', 'menu_crop': 'क्रॉप', 'menu_sheet': 'शीट', 'menu_download': '⬇️ डाउनलोड', 'menu_compare': 'तुलना', 'menu_guides': 'गाइड लाइनें', 'menu_fix_auto': 'अपने आप ठीक करें', 'back': '« वापस', 'reset': 'रीसेट', 'undo': '↩️ पूर्ववत', 'redo': '↪️ फिर से', 'size_title': 'फोटो साइज़ चुनें:', 'size_custom_prompt': 'साइज़ भेजें, जैसे <code>35x45 mm</code>, <code>2x2 in</code> या <code>413x531 px</code>।', 'size_bad': 'यह साइज़ समझ नहीं आया। जैसे <code>35x45 mm</code> या <code>2x2 in</code> लिखें।', 'bg_title': 'बैकग्राउंड रंग चुनें:', 'bg_custom_prompt': 'HEX रंग कोड भेजें, जैसे <code>#FF5733</code> — या कोई फोटो भेजें, मैं उसका औसत रंग लूँगा।', 'bg_bad_hex': 'यह सही HEX रंग नहीं है। उदाहरण: <code>#4A90D9</code>', 'bg_transparent': 'पारदर्शी (PNG)', 'enhance_title': 'सुधार — − / + दबाकर बदलें। हर टैप पर प्रीव्यू अपडेट होगा।', 'enh_glow': 'ग्लो', 'enh_exposure': 'एक्सपोज़र', 'enh_brightness': 'चमक', 'enh_contrast': 'कंट्रास्ट', 'enh_saturation': 'संतृप्ति', 'enh_warmth': 'गर्माहट', 'enh_sharpness': 'तीक्ष्णता', 'enh_smooth': 'त्वचा स्मूदिंग', 'enh_denoise': 'शोर कम करें', 'enh_look': 'लुक', 'look_natural': 'नैचुरल', 'look_bright': 'ब्राइट', 'look_studio': 'स्टूडियो', 'look_bw': 'ब्लैक एंड व्हाइट', 'enh_auto': 'ऑटो एन्हांस', 'enh_boost': 'क्वालिटी बूस्ट', 'crop_title': 'क्रॉप — खिसकाएँ, ज़ूम, घुमाएँ। साइज़ {size} तय है।', 'crop_step': 'कदम: {step}', 'crop_autofit': 'चेहरा ऑटो-फिट', 'sheet_title': 'शीट — पेपर, कॉपी और लेआउट।', 'sheet_paper': 'पेपर: {paper}', 'sheet_copies': 'कॉपी: {copies}', 'sheet_maxfit': 'अधिकतम फिट', 'sheet_margins': 'मार्जिन: {m} mm', 'sheet_gap': 'गैप: {g} mm', 'sheet_cuts_on': 'कट मार्क: चालू', 'sheet_cuts_off': 'कट मार्क: बंद', 'sheet_border_on': 'बॉर्डर: चालू', 'sheet_border_off': 'बॉर्डर: बंद', 'sheet_type_number': 'या कॉपी की संख्या सीधे लिखें।', 'copies_overflow': '{paper} पर {n} कॉपी नहीं आतीं (अधिकतम {max})। अधिकतम फिट या बड़ा पेपर चुनें।', 'download_done': 'हो गया ✅\nप्रिंट टिप: <b>100% / Actual size</b> पर प्रिंट करें, स्केलिंग नहीं।', 'target_kb_title': 'ऑनलाइन फ़ॉर्म के लिए फाइल साइज़:', 'target_kb_done': '{kb} KB पर सहेजा गया।', 'compare_caption': 'मूल (बाएँ) बनाम संसाधित (दाएँ)', 'session_cleared': 'सत्र साफ़ हो गया। तैयार होने पर नई फोटो भेजें।', 'reset_done': 'सारे बदलाव ऑटोमैटिक पर रीसेट।', 'session_expired': 'वह सत्र समाप्त हो गया। जारी रखने के लिए फोटो दोबारा भेजें।', 'resume_or_new': 'फिर से स्वागत है। पिछला सत्र जारी रखें या नई फोटो भेजें।', 'resume': 'जारी रखें', 'retry': 'फिर कोशिश करें', 'error_generic': 'अंतिम चरण में प्रोसेसिंग विफल रही। Retry दबाएँ या फोटो दोबारा भेजें।', 'settings_title': 'सेटिंग्स', 'set_dpi': 'एक्सपोर्ट DPI: {dpi}', 'set_delete_data': 'मेरा डेटा मिटाएँ', 'set_preset_save': 'मेरे प्रीसेट में सहेजें', 'set_preset_use': 'मेरा प्रीसेट लगाएँ', 'preset_saved': 'प्रीसेट सहेजा गया।', 'preset_applied': 'प्रीसेट लगाया गया।', 'no_preset': 'अभी कोई प्रीसेट नहीं है।', 'data_deleted': 'आपका सारा डेटा और फाइलें मिटा दी गईं।', 'admin_only': 'केवल एडमिन।', 'admin_stats': 'उपयोगकर्ता: {users}\nआज की जॉब: {jobs}\nआज की त्रुटियाँ: {errors}\nऔसत समय: {avg}s', 'admin_health': 'ठीक है। कतार: {q}, अस्थायी फाइलें: {t}', 'broadcast_prompt': 'सभी उपयोगकर्ताओं को भेजने के लिए संदेश लिखें।', 'broadcast_done': '{n} उपयोगकर्ताओं को संदेश भेजा गया।', 'cancel': 'रद्द किया गया।', 'bg_white': 'सफ़ेद', 'bg_offwhite': 'ऑफ़-व्हाइट', 'bg_lightgrey': 'हल्का ग्रे', 'bg_lightblue': 'हल्का नीला', 'bg_skyblue': 'आसमानी', 'bg_red': 'लाल', 'bg_yellow': 'पीला', 'bg_green': 'हरा', 'bg_black': 'काला', 'bg_custom': 'अपना रंग…', 'size_in_passport': 'भारतीय पासपोर्ट 35×45', 'size_in_visa': 'भारतीय वीज़ा / OCI 51×51', 'size_pan': 'PAN कार्ड 25×35', 'size_stamp': 'स्टाम्प 20×25', 'size_us': 'US पासपोर्ट 2×2 in', 'size_schengen': 'UK / शेंगेन 35×45', 'size_custom': 'कस्टम…', 'restore_more': 'ज़्यादा वापस लाएँ', 'erase_more': 'ज़्यादा मिटाएँ', 'feather': 'किनारे की नरमी', 'edge_shift': 'किनारा अंदर/बाहर', 'bg_edge_title': 'किनारे की क्वालिटी — मास्क को नरम, बड़ा या छोटा करें।', 'text_strip_prompt': 'फोटो के नीचे पट्टी के लिए टेक्स्ट भेजें (नाम और तारीख), या वापस दबाएँ।', 'delete_data_confirm': 'इससे आपकी सेटिंग्स, प्रीसेट और सत्र मिट जाएँगे। पक्का करने के लिए फिर दबाएँ।', 'album_processing': '{n} फोटो संसाधित हो रही हैं — एक क्षण…'},
}
def load_locales() -> None:
    """Kept as a startup hook for compatibility; locales are embedded above."""
    return None

def t(lang: str, key: str, **kw) -> str:
    text = _LOCALES.get(lang, _LOCALES[DEFAULT_LANG]).get(key)
    if text is None:
        text = _LOCALES[DEFAULT_LANG].get(key, key)
    try:
        return text.format(**kw) if kw else text
    except (KeyError, ValueError):
        return text

# ============================== SQLite storage ==============================

# ---------------- bot/services/storage.py ----------------

import json
import time
from typing import Any, Optional

import aiosqlite


_db: Optional[aiosqlite.Connection] = None

DEFAULT_RECIPE = {
    "exposure": 0, "brightness": 0, "contrast": 0, "saturation": 0,
    "warmth": 0, "sharpness": 0, "glow": 0, "smooth": 0, "denoise": 0,
    "look": "natural",
}
DEFAULT_CROP = {"dx": 0.0, "dy": 0.0, "zoom": 1.0, "rotate": 0.0,
                "flip": False, "step": "normal", "guides": False}


def default_session() -> dict:
    return {
        "orig": None, "work": None, "mask": None,
        "size": "in_passport", "custom_mm": None,
        "bg": "#FFFFFF", "transparent": False,
        "edge": {"feather": 2, "shift": 0},
        "recipe": dict(DEFAULT_RECIPE), "crop": dict(DEFAULT_CROP),
        "history": [], "hpos": -1,
        "copies": 8, "paper": "a4", "margin": 5.0, "gap": 2.0,
        "cut_marks": False, "border": False, "text_strip": None,
        "card_msg": None, "sheet_msg": None, "dpi": None,
    }


async def init() -> None:
    global _db
    _db = await aiosqlite.connect(config.DB_PATH)
    await _db.executescript("""
    CREATE TABLE IF NOT EXISTS users(
      user_id INTEGER PRIMARY KEY, name TEXT, lang TEXT DEFAULT NULL,
      dpi INTEGER DEFAULT 300, preset TEXT DEFAULT NULL, created REAL);
    CREATE TABLE IF NOT EXISTS sessions(
      user_id INTEGER PRIMARY KEY, data TEXT, updated REAL);
    CREATE TABLE IF NOT EXISTS stats(
      day TEXT PRIMARY KEY, jobs INTEGER DEFAULT 0,
      errors INTEGER DEFAULT 0, total_ms INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS file_cache(key TEXT PRIMARY KEY, file_id TEXT);
    """)
    await _db.commit()


async def close() -> None:
    if _db:
        await _db.close()


async def upsert_user(uid: int, name: str) -> None:
    await _db.execute(
        "INSERT INTO users(user_id,name,created) VALUES(?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name",
        (uid, name, time.time()))
    await _db.commit()


async def get_lang(uid: int) -> str:
    cur = await _db.execute("SELECT lang FROM users WHERE user_id=?", (uid,))
    row = await cur.fetchone()
    return (row[0] if row and row[0] else None) or config.DEFAULT_LANG


async def set_lang(uid: int, lang: str) -> None:
    await _db.execute("UPDATE users SET lang=? WHERE user_id=?", (lang, uid))
    await _db.commit()


async def get_dpi(uid: int) -> int:
    cur = await _db.execute("SELECT dpi FROM users WHERE user_id=?", (uid,))
    row = await cur.fetchone()
    return row[0] if row else 300


async def set_dpi(uid: int, dpi: int) -> None:
    await _db.execute("UPDATE users SET dpi=? WHERE user_id=?", (dpi, uid))
    await _db.commit()


async def get_session(uid: int) -> dict:
    cur = await _db.execute("SELECT data FROM sessions WHERE user_id=?", (uid,))
    row = await cur.fetchone()
    if not row:
        return default_session()
    s = default_session()
    s.update(json.loads(row[0]))
    return s


async def save_session(uid: int, data: dict) -> None:
    await _db.execute(
        "INSERT INTO sessions(user_id,data,updated) VALUES(?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
        (uid, json.dumps(data), time.time()))
    await _db.commit()


async def clear_session(uid: int) -> None:
    await _db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
    await _db.commit()


async def save_preset(uid: int, preset: dict) -> None:
    await _db.execute("UPDATE users SET preset=? WHERE user_id=?",
                      (json.dumps(preset), uid))
    await _db.commit()


async def get_preset(uid: int) -> Optional[dict]:
    cur = await _db.execute("SELECT preset FROM users WHERE user_id=?", (uid,))
    row = await cur.fetchone()
    return json.loads(row[0]) if row and row[0] else None


async def delete_user_data(uid: int) -> None:
    await _db.execute("DELETE FROM users WHERE user_id=?", (uid,))
    await _db.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
    await _db.commit()


async def bump_stat(jobs: int = 0, errors: int = 0, ms: int = 0) -> None:
    day = time.strftime("%Y-%m-%d")
    await _db.execute(
        "INSERT INTO stats(day,jobs,errors,total_ms) VALUES(?,?,?,?) "
        "ON CONFLICT(day) DO UPDATE SET jobs=jobs+?, errors=errors+?, total_ms=total_ms+?",
        (day, jobs, errors, ms, jobs, errors, ms))
    await _db.commit()


async def stats_today() -> dict:
    day = time.strftime("%Y-%m-%d")
    cur = await _db.execute(
        "SELECT jobs,errors,total_ms FROM stats WHERE day=?", (day,))
    row = await cur.fetchone() or (0, 0, 0)
    cur2 = await _db.execute("SELECT COUNT(*) FROM users")
    users = (await cur2.fetchone())[0]
    avg = round(row[2] / row[0] / 1000, 1) if row[0] else 0
    return {"users": users, "jobs": row[0], "errors": row[1], "avg": avg}


async def all_user_ids() -> list[int]:
    cur = await _db.execute("SELECT user_id FROM users")
    return [r[0] for r in await cur.fetchall()]


async def cache_get(key: str) -> Optional[str]:
    cur = await _db.execute("SELECT file_id FROM file_cache WHERE key=?", (key,))
    row = await cur.fetchone()
    return row[0] if row else None


async def cache_put(key: str, file_id: str) -> None:
    await _db.execute(
        "INSERT OR REPLACE INTO file_cache(key,file_id) VALUES(?,?)", (key, file_id))
    await _db.commit()
storage = SimpleNamespace(**{k: globals()[k] for k in ['DEFAULT_RECIPE', 'default_session', 'init', 'close', 'upsert_user', 'get_lang', 'set_lang', 'get_dpi', 'set_dpi', 'get_session', 'save_session', 'clear_session', 'save_preset', 'get_preset', 'delete_user_data', 'bump_stat', 'stats_today', 'all_user_ids', 'cache_get', 'cache_put'] if k in globals()})

# ---------------- bot/services/queue.py ----------------

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable


_executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS,
                               thread_name_prefix="cv")
_global = asyncio.Semaphore(config.MAX_CONCURRENT_JOBS)
_user_locks: dict[int, asyncio.Lock] = {}
_pending = 0


def position_estimate() -> int:
    return max(0, _pending - config.MAX_CONCURRENT_JOBS)


def user_lock(uid: int) -> asyncio.Lock:
    return _user_locks.setdefault(uid, asyncio.Lock())


async def run_cpu(fn: Callable, *args, **kwargs) -> Any:
    global _pending
    _pending += 1
    try:
        async with _global:
            loop = asyncio.get_running_loop()
            call = functools.partial(fn, *args, **kwargs)
            return await loop.run_in_executor(_executor, call)
    finally:
        _pending -= 1


def shutdown() -> None:
    _executor.shutdown(wait=False)
queue = SimpleNamespace(**{k: globals()[k] for k in ['position_estimate', 'user_lock', 'run_cpu', 'shutdown'] if k in globals()})

# ---------------- bot/services/face.py ----------------

import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

_mp = None
try:  # optional heavy dependency
    import mediapipe as mp  # noqa: F401
    _mp = mp
except Exception:
    _mp = None


def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def detect_faces(img: Image.Image) -> list[tuple[int, int, int, int]]:
    cv_img = _to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    if _mp is not None:
        try:
            with _mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5) as fd:
                res = fd.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            if res.detections:
                h, w = gray.shape
                boxes = []
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    boxes.append((int(bb.xmin * w), int(bb.ymin * h),
                                  int(bb.width * w), int(bb.height * h)))
                return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        except Exception:
            pass
    found = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in found]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def eye_line_angle(img: Image.Image) -> Optional[float]:
    gray = cv2.cvtColor(_to_cv(img), cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    roi = gray[y:y + int(h * 0.6), x:x + w]
    eyes = _eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(15, 15))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    c1 = (x1 + w1 / 2, y1 + h1 / 2)
    c2 = (x2 + w2 / 2, y2 + h2 / 2)
    return math.degrees(math.atan2(c2[1] - c1[1], c2[0] - c1[0]))


def straighten(img: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 0.4 or abs(angle) > 25:
        return img
    return img.rotate(-angle, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255))


def draw_numbered(img: Image.Image, boxes) -> Image.Image:
    out = img.convert("RGB").copy()
    arr = _to_cv(out)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(arr, (x, y), (x + w, y + h), (0, 200, 0), 3)
        cv2.putText(arr, str(i), (x + 6, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 200, 0), 4)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
facesvc = SimpleNamespace(**{k: globals()[k] for k in ['_to_cv', 'detect_faces', 'eye_line_angle', 'straighten', 'draw_numbered'] if k in globals()})

# ---------------- bot/services/background.py ----------------

import cv2
import numpy as np
from PIL import Image

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session
        # u2net_human_seg is tuned for people; isnet is a good fallback
        try:
            _session = new_session("u2net_human_seg")
        except Exception:
            _session = new_session("isnet-general-use")
    return _session


def remove(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    try:
        from rembg import remove as rembg_remove
        out = rembg_remove(img, session=_get_session())
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        mask = out.getchannel("A")
        return out, mask
    except Exception:
        mask = Image.new("L", img.size, 255)
        return img.convert("RGBA").putalpha(mask) or _with_alpha(img, mask)


def _with_alpha(img: Image.Image, mask: Image.Image) -> Image.Image:
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def refine_mask(mask: Image.Image, feather: int = 2, shift: int = 0) -> Image.Image:
    a = np.asarray(mask, dtype=np.uint8)
    if shift != 0:
        k = np.ones((3, 3), np.uint8)
        n = abs(shift)
        a = cv2.dilate(a, k, iterations=n) if shift > 0 else cv2.erode(a, k, iterations=n)
    if feather > 0:
        r = feather * 2 + 1
        a = cv2.GaussianBlur(a, (r, r), 0)
    # smooth jaggies
    a = cv2.medianBlur(a, 3)
    return Image.fromarray(a)


def decontaminate(img: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = np.asarray(img.convert("RGBA"), dtype=np.float32)
    a = np.asarray(mask, dtype=np.float32) / 255.0
    edge = (a > 0.05) & (a < 0.95)
    if not edge.any():
        return img
    blur = cv2.GaussianBlur(rgba[..., :3], (9, 9), 0)
    rgb = rgba[..., :3]
    # desaturate edge slightly and blend inward colour
    grey = rgb.mean(axis=2, keepdims=True)
    fixed = np.where(edge[..., None], rgb * 0.4 + grey * 0.2 + blur * 0.4, rgb)
    out = np.concatenate([fixed, rgba[..., 3:]], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def apply_background(cutout: Image.Image, mask: Image.Image,
                     color_hex: str | None, bg_image: Image.Image | None = None,
                     transparent: bool = False) -> Image.Image:
    w, h = cutout.size
    if transparent:
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    elif bg_image is not None:
        base = bg_image.convert("RGBA").resize((w, h))
    else:
        rgb = hex_to_rgb(color_hex or "#FFFFFF")
        base = Image.new("RGBA", (w, h), rgb + (255,))
    fg = cutout.convert("RGBA")
    fg.putalpha(mask)
    base.alpha_composite(fg)
    return base


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def valid_hex(s: str) -> bool:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return False
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def average_colour(img: Image.Image) -> str:
    small = img.convert("RGB").resize((1, 1))
    r, g, b = small.getpixel((0, 0))
    return f"#{r:02X}{g:02X}{b:02X}"


BG_PRESETS = {
    "bg_white": "#FFFFFF", "bg_offwhite": "#FAF9F6", "bg_lightgrey": "#D3D3D3",
    "bg_lightblue": "#BFE3F0", "bg_skyblue": "#87CEEB", "bg_red": "#D32F2F",
    "bg_yellow": "#FDD835", "bg_green": "#388E3C", "bg_black": "#111111",
}
bgsvc = SimpleNamespace(**{k: globals()[k] for k in ['_get_session', 'remove', '_with_alpha', 'refine_mask', 'decontaminate', 'apply_background', 'hex_to_rgb', 'valid_hex', 'average_colour', 'BG_PRESETS'] if k in globals()})

# ---------------- bot/services/enhance.py ----------------

import cv2
import numpy as np
from PIL import Image, ImageEnhance

LOOKS = {
    "natural": {},
    "bright": {"brightness": 2, "contrast": 1, "saturation": 1},
    "studio": {"contrast": 2, "sharpness": 2, "warmth": -1},
    "bw": {"saturation": -10},
}


def apply_recipe(img: Image.Image, r: dict) -> Image.Image:
    out = img.convert("RGB")
    look = LOOKS.get(r.get("look", "natural"), {})
    merged = dict(r)
    for k, v in look.items():
        merged[k] = merged.get(k, 0) + v

    if merged.get("denoise", 0) > 0:
        arr = np.asarray(out)
        strength = min(10, 2 + merged["denoise"])
        arr = cv2.fastNlMeansDenoisingColored(arr, None, strength, strength, 7, 21)
        out = Image.fromarray(arr)

    def adj(factor_base: float, step: int) -> float:
        return max(0.0, 1.0 + factor_base * step)

    b = merged.get("brightness", 0) + merged.get("exposure", 0)
    if b:
        out = ImageEnhance.Brightness(out).enhance(adj(0.06, b))
    if merged.get("contrast"):
        out = ImageEnhance.Contrast(out).enhance(adj(0.07, merged["contrast"]))
    if merged.get("saturation"):
        out = ImageEnhance.Color(out).enhance(adj(0.10, merged["saturation"]))

    if merged.get("warmth"):
        arr = np.asarray(out).astype(np.float32)
        shift = merged["warmth"] * 6.0
        arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] - shift, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8))

    if merged.get("smooth", 0) > 0:  # skin smoothing that keeps texture
        arr = np.asarray(out)
        n = merged["smooth"]
        smooth = cv2.bilateralFilter(arr, 9, 25 + 15 * n, 25 + 15 * n)
        alpha = min(0.5, 0.12 * n)
        out = Image.fromarray(cv2.addWeighted(arr, 1 - alpha, smooth, alpha, 0))

    if merged.get("glow", 0) > 0:
        n = merged["glow"]
        blur = out.filter(_gauss(6 + 2 * n))
        out = Image.blend(out, blur, min(0.25, 0.05 * n))
        out = ImageEnhance.Brightness(out).enhance(1.0 + 0.01 * n)

    if merged.get("sharpness"):
        out = ImageEnhance.Sharpness(out).enhance(adj(0.12, merged["sharpness"]))
    return out


def _gauss(radius: float):
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(radius)


def auto_recipe(img: Image.Image) -> dict:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    r: dict = {"denoise": 1, "sharpness": 1, "contrast": 1, "glow": 0,
               "smooth": 0, "look": "natural", "warmth": 0,
               "brightness": 0, "saturation": 0, "exposure": 0}
    if mean < 90:
        r["brightness"] = 3
    elif mean < 120:
        r["brightness"] = 1
    elif mean > 210:
        r["brightness"] = -2
    return r


def quality_boost(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    arr = cv2.fastNlMeansDenoisingColored(arr, None, 4, 4, 7, 21)
    if min(arr.shape[:2]) < 600:
        arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(arr, (0, 0), 2.0)
    arr = cv2.addWeighted(arr, 1.4, blur, -0.4, 0)
    return Image.fromarray(arr)


def even_lighting(img: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr[..., 0] = clahe.apply(arr[..., 0])
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))
ensvc = SimpleNamespace(**{k: globals()[k] for k in ['LOOKS', 'apply_recipe', '_gauss', 'auto_recipe', 'quality_boost', 'even_lighting'] if k in globals()})

# ---------------- bot/services/layout.py ----------------

import json
from typing import Optional

from PIL import Image, ImageDraw


MM_PER_IN = 25.4

SIZES: dict[str, tuple[float, float]] = {
    "in_passport": (35, 45), "in_visa": (51, 51), "pan": (25, 35),
    "stamp": (20, 25), "us": (2 * MM_PER_IN, 2 * MM_PER_IN),
    "schengen": (35, 45),
}
PAPERS: dict[str, tuple[float, float]] = {
    "a4": (210, 297), "a3": (297, 420), "a5": (148, 210),
    "letter": (8.5 * MM_PER_IN, 11 * MM_PER_IN),
    "4x6": (4 * MM_PER_IN, 6 * MM_PER_IN), "5x7": (5 * MM_PER_IN, 7 * MM_PER_IN),
}


def mm_to_px(mm: float, dpi: int) -> int:
    return round(mm * dpi / MM_PER_IN)


def photo_px(mm: tuple[float, float], dpi: int) -> tuple[int, int]:
    return mm_to_px(mm[0], dpi), mm_to_px(mm[1], dpi)


def load_template() -> Optional[dict]:
    try:
        return json.loads(config.TEMPLATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "name": "Indian Passport A4 Sheet",
            "paper": "a4",
            "paper_mm": [210, 297],
            "photo_mm": [35, 45],
            "dpi": 300,
            "margin_top_mm": 10.0,
            "margin_left_mm": 14.8,
            "margin_right_mm": 14.8,
            "margin_bottom_mm": 10.0,
            "gap_x_mm": 2.0,
            "gap_y_mm": 2.0,
            "border_mm": 0.4,
            "border_color": "#000000",
            "cut_marks": False,
        }


def size_mm(sess: dict) -> tuple[float, float]:
    if sess.get("size") == "custom" and sess.get("custom_mm"):
        return tuple(sess["custom_mm"])  # type: ignore
    if sess.get("size") == "template":
        tpl = load_template()
        if tpl:
            return tuple(tpl["photo_mm"])  # type: ignore
    return SIZES.get(sess.get("size", "in_passport"), SIZES["in_passport"])


def paper_mm(sess: dict) -> tuple[float, float]:
    return PAPERS.get(sess.get("paper", "a4"), PAPERS["a4"])


def grid(paper: tuple[float, float], photo: tuple[float, float],
         margin: float, gap: float) -> tuple[int, int, float, float]:
    pw, ph = paper
    fw, fh = photo
    cols = int((pw - 2 * margin + gap) // (fw + gap))
    rows = int((ph - 2 * margin + gap) // (fh + gap))
    cols, rows = max(cols, 0), max(rows, 0)
    used_w = cols * fw + (cols - 1) * gap
    used_h = rows * fh + (rows - 1) * gap
    x0 = (pw - used_w) / 2 if cols else margin
    y0 = (ph - used_h) / 2 if rows else margin
    return cols, rows, x0, y0


def max_fit(paper, photo, margin, gap) -> int:
    cols, rows, _, _ = grid(paper, photo, margin, gap)
    return cols * rows


def placements(paper, photo, margin, gap, copies: int) -> list[tuple[float, float]]:
    cols, rows, x0, y0 = grid(paper, photo, margin, gap)
    out = []
    for r in range(rows):
        for c in range(cols):
            if len(out) >= copies:
                return out
            out.append((x0 + c * (photo[0] + gap), y0 + r * (photo[1] + gap)))
    return out


def render_sheet(photo_img: Image.Image, sess: dict, dpi: int = 300,
                 preview: bool = False) -> tuple[Image.Image, int]:
    paper = paper_mm(sess)
    photo = size_mm(sess)
    margin, gap = sess.get("margin", 5.0), sess.get("gap", 2.0)
    dpi = 96 if preview else dpi
    sheet = Image.new("RGB", (mm_to_px(paper[0], dpi), mm_to_px(paper[1], dpi)),
                      "white")
    draw = ImageDraw.Draw(sheet)
    ph = photo_px(photo, dpi)
    pic = photo_img.convert("RGB").resize(ph, Image.LANCZOS)
    if sess.get("text_strip") and not preview:
        strip = mm_to_px(4, dpi)
        canvas = Image.new("RGB", (ph[0], ph[1] + strip), "white")
        canvas.paste(pic, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, ph[1] + 2), sess["text_strip"][:60], fill="black")
        pic = canvas
        photo = (photo[0], photo[1] + 4)

    pos = placements(paper, photo, margin, gap, sess.get("copies", 8))
    border_px = max(1, mm_to_px(0.4, dpi))
    for x_mm, y_mm in pos:
        x, y = mm_to_px(x_mm, dpi), mm_to_px(y_mm, dpi)
        sheet.paste(pic, (x, y))
        if sess.get("border"):
            draw.rectangle([x, y, x + pic.width - 1, y + pic.height - 1],
                           outline="black", width=border_px)
        if sess.get("cut_marks"):
            cm = mm_to_px(3, dpi)
            for cx, cy, dx, dy in ((x, y, -1, -1), (x + pic.width, y, 1, -1),
                                   (x, y + pic.height, -1, 1),
                                   (x + pic.width, y + pic.height, 1, 1)):
                draw.line([cx, cy, cx + dx * cm, cy], fill="black", width=1)
                draw.line([cx, cy, cx, cy + dy * cm], fill="black", width=1)
    if preview:  # mm rulers
        step = mm_to_px(10, dpi)
        for x in range(0, sheet.width, step):
            draw.line([x, 0, x, 6], fill="#999")
        for y in range(0, sheet.height, step):
            draw.line([0, y, 6, y], fill="#999")
    return sheet, len(pos)
layout = SimpleNamespace(**{k: globals()[k] for k in ['MM_PER_IN', 'SIZES', 'PAPERS', 'mm_to_px', 'photo_px', 'load_template', 'size_mm', 'paper_mm', 'grid', 'max_fit', 'placements', 'render_sheet'] if k in globals()})

# ---------------- bot/services/pdf_export.py ----------------

import io

from PIL import Image
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas



def sheet_pdf(photo_img: Image.Image, sess: dict) -> bytes:
    paper = layout.paper_mm(sess)
    photo = layout.size_mm(sess)
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(paper[0] * mm, paper[1] * mm))
    pos = layout.placements(paper, photo, sess.get("margin", 5.0),
                            sess.get("gap", 2.0), sess.get("copies", 8))
    reader = ImageReader(photo_img.convert("RGB"))
    for x_mm, y_mm in pos:
        # reportlab origin is bottom-left
        c.drawImage(reader, x_mm * mm,
                    (paper[1] - y_mm - photo[1]) * mm,
                    width=photo[0] * mm, height=photo[1] * mm)
        if sess.get("border"):
            c.setLineWidth(0.4 * mm)
            c.rect(x_mm * mm, (paper[1] - y_mm - photo[1]) * mm,
                   photo[0] * mm, photo[1] * mm)
        if sess.get("cut_marks"):
            c.setLineWidth(0.2 * mm)
            for cx, cy in ((x_mm, y_mm), (x_mm + photo[0], y_mm),
                           (x_mm, y_mm + photo[1]),
                           (x_mm + photo[0], y_mm + photo[1])):
                ry = paper[1] - cy
                c.line((cx - 3) * mm, ry * mm, (cx + 3) * mm, ry * mm)
                c.line(cx * mm, (ry - 3) * mm, cx * mm, (ry + 3) * mm)
    c.showPage()
    c.save()
    return buf.getvalue()


def to_jpeg_target_kb(img: Image.Image, target_kb: int,
                      tolerance: float = 0.15) -> tuple[bytes, int]:
    lo, hi = 20, 95
    best = b""
    work = img.convert("RGB")
    for _ in range(8):
        q = (lo + hi) // 2
        buf = io.BytesIO()
        work.save(buf, "JPEG", quality=q, optimize=True)
        kb = buf.tell() // 1024
        best = buf.getvalue()
        if kb <= target_kb:
            if kb >= target_kb * (1 - tolerance):
                break
            lo = q + 1
        else:
            hi = q - 1
        if hi < lo:
            break
    kb = len(best) // 1024
    while kb > target_kb and min(work.size) > 400:  # shrink if quality wasn't enough
        work = work.resize((int(work.width * 0.85), int(work.height * 0.85)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        work.save(buf, "JPEG", quality=85, optimize=True)
        best, kb = buf.getvalue(), buf.tell() // 1024
    return best, kb
pdf_export = SimpleNamespace(**{k: globals()[k] for k in ['sheet_pdf', 'to_jpeg_target_kb'] if k in globals()})

# ---------------- bot/services/compliance.py ----------------

import numpy as np
from PIL import Image



def check(photo: Image.Image, face_box, mm_size: tuple[float, float],
          orig_px: tuple[int, int], dpi: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    w, h = photo.size
    if face_box:
        x, y, fw, fh = face_box
        out.append(("check_face", "ok"))
        head_ratio = fh / h
        # Indian passport: head ~32-36 mm of 45 mm => 0.71-0.80
        lo, hi = (0.66, 0.86) if mm_size[1] >= 40 else (0.55, 0.9)
        out.append(("check_head",
                    "ok" if lo <= head_ratio <= hi else
                    "warn" if lo - 0.1 <= head_ratio <= hi + 0.1 else "fail"))
        cx = x + fw / 2
        off = abs(cx - w / 2) / w
        out.append(("check_center", "ok" if off < 0.08 else "warn" if off < 0.15 else "fail"))
    else:
        out += [("check_face", "fail"), ("check_head", "fail"), ("check_center", "fail")]

    arr = np.asarray(photo.convert("RGB"), dtype=np.int16)
    gray = arr.mean(axis=2)
    # tilt: re-detect eyes on the crop
    eyes = facesvc.eye_line_angle(photo) if face_box else None
    if eyes is None:
        out.append(("check_tilt", "warn"))
    else:
        out.append(("check_tilt", "ok" if abs(eyes) < 3 else "warn" if abs(eyes) < 7 else "fail"))
    out.append(("check_eyes", "ok"))  # Haar-based; treat found face as eyes open

    # background uniformity: corners should be near-constant
    corners = np.concatenate([
        gray[: h // 10, : w // 10].ravel(), gray[: h // 10, -w // 10:].ravel(),
        gray[-h // 10:, : w // 10].ravel(), gray[-h // 10:, -w // 10:].ravel()])
    out.append(("check_bg", "ok" if corners.std() < 12 else "warn" if corners.std() < 30 else "fail"))

    mean = float(gray.mean())
    out.append(("check_bright", "ok" if 90 < mean < 220 else "warn" if 60 < mean < 240 else "fail"))

    need_w = int(mm_size[0] * dpi / 25.4)
    out.append(("check_res", "ok" if min(orig_px) >= need_w else "fail"))

    # face shadows: left/right half brightness difference in face region
    if face_box:
        x, y, fw, fh = face_box
        region = gray[y:y + fh, x:x + fw]
        if region.size:
            lh, rh = region[:, : region.shape[1] // 2].mean(), region[:, region.shape[1] // 2:].mean()
            d = abs(lh - rh)
            out.append(("check_shadow", "ok" if d < 18 else "warn" if d < 35 else "fail"))
        else:
            out.append(("check_shadow", "warn"))
    else:
        out.append(("check_shadow", "fail"))
    return out


ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌"}


def render_text(checks: list[tuple[str, str]], t_func) -> str:
    return "   ".join(f"{ICON[s]} {t_func(k)}" for k, s in checks)
compliance = SimpleNamespace(**{k: globals()[k] for k in ['check', 'ICON', 'render_text'] if k in globals()})

# ---------------- bot/services/pipeline.py ----------------

import time
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageOps


pillow_heif.register_heif_opener()

ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def load_photo(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.width * img.height > config.MAX_PIXELS:
        raise ValueError("image too large")
    return img.convert("RGB")


def working_copy(img: Image.Image, px: int = config.WORKING_PX) -> Image.Image:
    img = img.copy()
    img.thumbnail((px, px), Image.LANCZOS)
    return img


def auto_process(sess: dict, uid_dir: Path) -> dict:
    t0 = time.time()
    work = Image.open(sess["work"]).convert("RGB")

    angle = facesvc.eye_line_angle(work)
    if angle:
        work = facesvc.straighten(work, angle)

    boxes = facesvc.detect_faces(work)
    if not sess.get("face"):
        sess["face"] = boxes[0] if boxes else None

    if boxes:
        cutout, mask = bgsvc.remove(work)
        sess["has_bg_removed"] = True
    else:
        cutout, mask = work.convert("RGBA"), Image.new("L", work.size, 255)
        sess["has_bg_removed"] = False

    edge = sess.get("edge", {"feather": 2, "shift": 0})
    mask = bgsvc.refine_mask(mask, edge.get("feather", 2), edge.get("shift", 0))
    mask.save(uid_dir / "mask.png")
    sess["mask"] = str(uid_dir / "mask.png")
    cutout.convert("RGB").save(uid_dir / "cutout.png")
    sess["cutout"] = str(uid_dir / "cutout.png")
    work.save(uid_dir / "aligned.png")
    sess["aligned"] = str(uid_dir / "aligned.png")

    if not sess["history"]:
        auto = ensvc.auto_recipe(work)
        auto.update({k: v for k, v in sess["recipe"].items() if v not in (0, "natural")})
        sess["recipe"] = auto
        sess["history"] = [dict(auto)]
        sess["hpos"] = 0
    sess["elapsed"] = round(time.time() - t0, 2)
    return sess


def crop_to_standard(img: Image.Image, sess: dict,
                     face_box) -> Image.Image:
    mm = layout.size_mm(sess)
    aspect = mm[0] / mm[1]
    w, h = img.size
    crop = sess.get("crop", {})
    zoom = max(0.5, crop.get("zoom", 1.0))
    if face_box:
        x, y, fw, fh = face_box
        head_h = fh * (1.55 if mm[1] >= 40 else 1.3)
        ch = head_h / (0.75 * zoom)
        cw = ch * aspect
        cx = x + fw / 2 + crop.get("dx", 0) * w * 0.05
        cy = y + fh * 0.55 + crop.get("dy", 0) * h * 0.05
    else:
        ch = min(h, w / aspect) / zoom
        cw = ch * aspect
        cx, cy = w / 2, h * 0.45
    left = min(max(cx - cw / 2, 0), max(w - cw, 0))
    top = min(max(cy - ch / 2, 0), max(h - ch, 0))
    if cw > w:  # widen beyond image: scale down crop
        scale = w / cw
        cw, ch = w, ch * scale
        left = 0
    return img.crop((int(left), int(top), int(left + cw), int(top + ch)))


def render_preview(sess: dict, guides: bool = False) -> Image.Image:
    cutout = Image.open(sess["cutout"]).convert("RGBA")
    mask = Image.open(sess["mask"])
    face = sess.get("face")
    crop = sess.get("crop", {})
    if crop.get("rotate"):
        cutout = cutout.rotate(-crop["rotate"], resample=Image.BICUBIC)
        mask = mask.rotate(-crop["rotate"], resample=Image.BICUBIC)
    if crop.get("flip"):
        cutout = ImageOps.mirror(cutout)
        mask = ImageOps.mirror(mask)
    base = bgsvc.apply_background(
        cutout, mask, sess.get("bg"),
        transparent=sess.get("transparent", False))
    base = ensvc.apply_recipe(base.convert("RGB"), sess["recipe"])
    base = crop_to_standard(base, sess, face)
    px = layout.photo_px(layout.size_mm(sess), 300)
    base = base.resize((px[0] // 2, px[1] // 2), Image.LANCZOS)
    if guides or crop.get("guides"):
        d = ImageDraw.Draw(base)
        W, H = base.size
        d.line([W / 2, 0, W / 2, H], fill=(0, 200, 0), width=1)
        d.line([0, H * 0.62, W, H * 0.62], fill=(0, 150, 255), width=1)  # eyes
        d.line([0, H * 0.13, W, H * 0.13], fill=(255, 80, 80), width=1)  # crown
        d.line([0, H * 0.90, W, H * 0.90], fill=(255, 80, 80), width=1)  # chin
    return base


def render_final(sess: dict) -> Image.Image:
    uid_dir = Path(sess["work"]).parent
    orig = Image.open(sess["orig"]).convert("RGB")
    aligned_path = sess.get("aligned")
    scale = 1.0
    if aligned_path and Path(aligned_path).exists():
        work = Image.open(aligned_path)
        scale = orig.width / work.width
    # Re-run bg removal at full res only if not cached at high res
    cutout_full = uid_dir / "cutout_full.png"
    mask_full = uid_dir / "mask_full.png"
    if not cutout_full.exists():
        angle = facesvc.eye_line_angle(orig)
        if angle:
            orig = facesvc.straighten(orig, angle)
        cutout, mask = bgsvc.remove(orig)
        edge = sess.get("edge", {"feather": 2, "shift": 0})
        mask = bgsvc.refine_mask(mask, int(edge.get("feather", 2) * scale),
                                 int(edge.get("shift", 0) * scale))
        cutout.convert("RGB").save(cutout_full)
        mask.save(mask_full)
    cutout = Image.open(cutout_full).convert("RGBA")
    mask = Image.open(mask_full)
    crop = sess.get("crop", {})
    if crop.get("rotate"):
        cutout = cutout.rotate(-crop["rotate"], resample=Image.BICUBIC)
        mask = mask.rotate(-crop["rotate"], resample=Image.BICUBIC)
    if crop.get("flip"):
        cutout = ImageOps.mirror(cutout)
        mask = ImageOps.mirror(mask)
    base = bgsvc.apply_background(cutout, mask, sess.get("bg"),
                                  transparent=sess.get("transparent", False))
    base = ensvc.apply_recipe(base.convert("RGB"), sess["recipe"])
    face = sess.get("face")
    if face:
        face = [int(v * scale) for v in face]
    base = crop_to_standard(base, sess, face)
    dpi = sess.get("dpi") or 300
    base = base.resize(layout.photo_px(layout.size_mm(sess), dpi), Image.LANCZOS)
    return base


def compare_image(sess: dict) -> Image.Image:
    orig = Image.open(sess["work"]).convert("RGB")
    prev = render_preview(sess)
    h = max(orig.height, prev.height)
    o = orig.resize((int(orig.width * h / orig.height), h))
    p = prev.resize((int(prev.width * h / prev.height), h))
    out = Image.new("RGB", (o.width + p.width + 10, h), "white")
    out.paste(o, (0, 0))
    out.paste(p, (o.width + 10, 0))
    return out
pipeline = SimpleNamespace(**{k: globals()[k] for k in ['ALLOWED', 'load_photo', 'working_copy', 'auto_process', 'crop_to_standard', 'render_preview', 'render_final', 'compare_image'] if k in globals()})

# ---------------- bot/states.py ----------------

import json
from pathlib import Path

from aiogram.fsm.state import State, StatesGroup



class Flow(StatesGroup):
    waiting_language = State()
    waiting_custom_size = State()
    waiting_custom_colour = State()
    waiting_copies = State()
    waiting_text_strip = State()
    waiting_broadcast = State()
    waiting_face_pick = State()
    waiting_manual_crop = State()


_LOCALES: dict[str, dict] = {}


def load_locales() -> None:
    """Load external locale files when present, retaining embedded safe defaults.

    Locale files are deployment assets, but a missing optional locale must not
    prevent the bot from starting. Missing or malformed files fall back to the
    embedded dictionaries defined above.
    """
    locale_dir = Path(__file__).parent / "locale"
    for lang in config.LANGS:
        fallback = dict(_LOCALES.get(lang, {}))
        path = locale_dir / f"{lang}.json"
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("locale root must be an object")
            fallback.update(loaded)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger("bot").warning("Using embedded %s locale: %s", lang, exc)
        _LOCALES[lang] = fallback


def t(lang: str, key: str, **kw) -> str:
    s = _LOCALES.get(lang, _LOCALES[config.DEFAULT_LANG]).get(key)
    if s is None:
        s = _LOCALES[config.DEFAULT_LANG].get(key, key)
    return s.format(**kw) if kw else s

# ---------------- bot/keyboards.py ----------------

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


class Cb(CallbackData, prefix="c"):
    a: str          # action
    v: str = ""     # value


def btn(text: str, a: str, v: str = "") -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=Cb(a=a, v=v).pack())


def lang_kb() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn("English", "lang", "en"), btn("हिन्दी", "lang", "hi"))
    return b.as_markup()


def main_menu(t) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn(t("menu_size"), "menu", "size"), btn(t("menu_background"), "menu", "bg"))
    b.row(btn(t("menu_enhance"), "menu", "enh"), btn(t("menu_crop"), "menu", "crop"))
    b.row(btn(t("menu_compare"), "cmp"), btn(t("menu_guides"), "guides"))
    b.row(btn(t("menu_sheet"), "menu", "sheet"))
    b.row(btn(t("menu_download"), "menu", "dl"))
    return b.as_markup()


def size_menu(t, current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    keys = ["in_passport", "in_visa", "pan", "stamp", "us", "schengen"]
    for i in range(0, len(keys), 2):
        row = [btn(("✓ " if k == current else "") + t(f"size_{k}"), "size", k)
               for k in keys[i:i + 2]]
        b.row(*row)
    b.row(btn(t("size_custom"), "size", "custom"))
    b.row(btn(t("back"), "menu", "main"))
    return b.as_markup()


def bg_menu(t, current: str) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    keys = list(BG_PRESETS)
    for i in range(0, len(keys), 3):
        b.row(*[btn(("✓ " if BG_PRESETS[k] == current else "") + t(k), "bg", k)
                for k in keys[i:i + 3]])
    b.row(btn(t("bg_custom"), "bg", "custom"), btn(t("bg_transparent"), "bg", "transp"))
    b.row(btn(t("restore_more"), "mask", "restore"), btn(t("erase_more"), "mask", "erase"))
    b.row(btn(f'{t("feather")} −', "edge", "f-"), btn(f'{t("feather")} +', "edge", "f+"),
          btn(f'{t("edge_shift")} −', "edge", "s-"), btn(f'{t("edge_shift")} +', "edge", "s+"))
    b.row(btn(t("reset"), "bg", "reset"), btn(t("back"), "menu", "main"))
    return b.as_markup()


ENH_KEYS = ["glow", "exposure", "brightness", "contrast", "saturation",
            "warmth", "sharpness", "smooth", "denoise"]


def enhance_menu(t, recipe: dict) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    for k in ENH_KEYS:
        v = recipe.get(k, 0)
        b.row(btn(f'{t(f"enh_{k}")}  −', "enh", f"{k}-"),
              InlineKeyboardButton(text=f"{v}/10", callback_data=Cb(a="noop").pack()),
              btn("+", "enh", f"{k}+"))
    b.row(btn(t("look_natural"), "look", "natural"), btn(t("look_bright"), "look", "bright"),
          btn(t("look_studio"), "look", "studio"), btn(t("look_bw"), "look", "bw"))
    b.row(btn(t("enh_auto"), "enh", "auto"), btn(t("enh_boost"), "enh", "boost"))
    b.row(btn(t("undo"), "hist", "u"), btn(t("redo"), "hist", "r"), btn(t("reset"), "enh", "reset"))
    b.row(btn(t("back"), "menu", "main"))
    return b.as_markup()


def crop_menu(t, sess) -> InlineKeyboardMarkup:
    step = sess["crop"].get("step", "normal")
    b = InlineKeyboardBuilder()
    b.row(btn("←", "mv", "l"), btn("↑", "mv", "u"), btn("↓", "mv", "d"), btn("→", "mv", "r"))
    b.row(btn("Zoom −", "mv", "z-"), btn("Zoom +", "mv", "z+"))
    b.row(btn("−5°", "rot", "-5"), btn("−1°", "rot", "-1"),
          btn("+1°", "rot", "1"), btn("+5°", "rot", "5"))
    b.row(btn("Flip", "mv", "flip"), btn(t("crop_autofit"), "mv", "fit"),
          btn(t("crop_step", step=step), "mv", "step"))
    b.row(btn(t("undo"), "hist", "u"), btn(t("redo"), "hist", "r"),
          btn(t("reset"), "mv", "reset"))
    b.row(btn(t("back"), "menu", "main"))
    return b.as_markup()


def sheet_menu(t, sess, fits: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn(t("sheet_paper", paper=sess["paper"].upper()), "paper", "open"))
    b.row(btn("−", "cp", "-1"),
          InlineKeyboardButton(text=t("sheet_copies", copies=sess["copies"]),
                               callback_data=Cb(a="noop").pack()),
          btn("+", "cp", "+1"))
    b.row(*[btn(str(n), "cp", str(n)) for n in (2, 4, 6, 8)])
    b.row(*[btn(str(n), "cp", str(n)) for n in (12, 16, 24)],
          btn(t("sheet_maxfit"), "cp", "max"))
    b.row(btn(t("sheet_margins", m=sess["margin"]), "lay", "m"),
          btn(t("sheet_gap", g=sess["gap"]), "lay", "g"))
    b.row(btn(t("sheet_cuts_on" if sess["cut_marks"] else "sheet_cuts_off"), "lay", "cuts"),
          btn(t("sheet_border_on" if sess["border"] else "sheet_border_off"), "lay", "border"))
    b.row(btn(t("menu_download"), "dl", "sheet"))
    b.row(btn(t("back"), "menu", "main"))
    return b.as_markup()


def paper_menu(t) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    keys = ["a4", "a3", "a5", "letter", "4x6", "5x7"]
    for i in range(0, len(keys), 3):
        b.row(*[btn(k.upper(), "paper", k) for k in keys[i:i + 3]])
    b.row(btn(t("back"), "menu", "sheet"))
    return b.as_markup()


def download_menu(t) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn("PDF (A4 sheet)", "dl", "pdf"), btn("PNG (sheet)", "dl", "png"))
    b.row(btn("JPG (sheet)", "dl", "jpg"), btn("Single photo", "dl", "single"))
    b.row(btn("<50 KB", "tkb", "50"), btn("<100 KB", "tkb", "100"), btn("<200 KB", "tkb", "200"))
    b.row(btn(t("back"), "menu", "main"))
    return b.as_markup()


def settings_menu(t, dpi: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn(t("set_dpi", dpi=dpi), "dpi", "toggle"))
    b.row(btn(t("set_preset_save"), "preset", "save"),
          btn(t("set_preset_use"), "preset", "use"))
    b.row(btn(t("set_delete_data"), "deldata", "ask"))
    return b.as_markup()


def retry_kb(t) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn(t("retry"), "retry"))
    return b.as_markup()


def faces_kb(n: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(*[btn(str(i), "face", str(i - 1)) for i in range(1, min(n, 8) + 1)])
    return b.as_markup()


def noface_kb(t) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.row(btn(t("manual_crop"), "face", "manual"))
    return b.as_markup()

# ---------------- bot/middlewares.py ----------------

import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject



class FloodMiddleware(BaseMiddleware):
    def __init__(self) -> None:
        self._hits: dict[int, deque] = defaultdict(deque)

    async def __call__(self, handler, event: TelegramObject, data: dict) -> Any:
        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)
        now = time.monotonic()
        q = self._hits[user.id]
        while q and now - q[0] > config.FLOOD_WINDOW_S:
            q.popleft()
        if len(q) >= config.FLOOD_MAX_EVENTS:
            if isinstance(event, CallbackQuery):
                await event.answer()
            return None
        q.append(now)
        return await handler(event, data)


class UserMiddleware(BaseMiddleware):

    async def __call__(self, handler, event: TelegramObject, data: dict) -> Any:
        user = data.get("event_from_user")
        if user is not None:
            await storage.upsert_user(user.id, user.full_name)
            data["lang"] = await storage.get_lang(user.id)
        else:
            data["lang"] = config.DEFAULT_LANG
        return await handler(event, data)


class ErrorMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: TelegramObject, data: dict) -> Any:
        try:
            return await handler(event, data)
        except Exception:
            import logging, traceback
            logging.getLogger("bot").exception("handler error")
            await storage.bump_stat(errors=1)
            bot = data.get("bot")
            lang = data.get("lang", config.DEFAULT_LANG)
            try:
                if isinstance(event, CallbackQuery):
                    await event.answer(t(lang, "error_generic"), show_alert=True)
                elif isinstance(event, Message):
                    await event.answer(t(lang, "error_generic"))
                if bot and config.ADMIN_IDS:
                    tb = traceback.format_exc()[-3500:]
                    for aid in config.ADMIN_IDS:
                        await bot.send_message(aid, f"⚠️ Handler error:\n<pre>{tb}</pre>")
            except Exception:
                pass
            return None

router = Router()

# ---------------- bot/handlers/core.py ----------------

import shutil
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message





@router.message(CommandStart())
async def cmd_start(m: Message, state: FSMContext, lang: str):
    await storage.upsert_user(m.from_user.id, m.from_user.full_name)
    cur = await storage.get_lang(m.from_user.id)
    # First-ever start: language pick. After that: welcome directly.
    sess = await storage.get_session(m.from_user.id)
    if not sess.get("card_msg") and cur == config.DEFAULT_LANG:
        await state.set_state(Flow.waiting_language)
        await m.answer(t("en", "choose_language"), reply_markup=lang_kb())
    else:
        await m.answer(t(cur, "welcome"))
        await m.answer(t(cur, "privacy_note", ttl=config.TEMP_TTL_MINUTES))


@router.callback_query(Cb.filter(F.a == "lang"))
async def lang_pick(cb, state: FSMContext):
    data = Cb.unpack(cb.data)
    await storage.set_lang(cb.from_user.id, data.v)
    await state.clear()
    lang = data.v
    await cb.message.edit_text(t(lang, "welcome"))
    await cb.message.answer(t(lang, "privacy_note", ttl=config.TEMP_TTL_MINUTES))
    await cb.answer()


@router.message(Command("help"))
async def cmd_help(m: Message, lang: str):
    await m.answer(t(lang, "help"))


@router.message(Command("language"))
async def cmd_language(m: Message, state: FSMContext):
    await state.set_state(Flow.waiting_language)
    await m.answer(t("en", "choose_language"), reply_markup=lang_kb())


@router.message(Command("privacy"))
async def cmd_privacy(m: Message, lang: str):
    await m.answer(t(lang, "privacy_full", ttl=config.TEMP_TTL_MINUTES))


@router.message(Command("reset"))
async def cmd_reset(m: Message, lang: str):
    sess = await storage.get_session(m.from_user.id)
    d = storage.default_session()
    for k in ("orig", "work", "mask", "cutout", "aligned", "face",
              "card_msg", "sheet_msg"):
        d[k] = sess.get(k)
    d["copies"], d["paper"] = sess["copies"], sess["paper"]
    await storage.save_session(m.from_user.id, d)
    await m.answer(t(lang, "reset_done"))


@router.message(Command("cancel"))
async def cmd_cancel(m: Message, state: FSMContext, lang: str):
    await state.clear()
    await _wipe_files(m.from_user.id)
    await storage.clear_session(m.from_user.id)
    await m.answer(t(lang, "session_cleared"))


async def _wipe_files(uid: int) -> None:
    d = config.TEMP_DIR / str(uid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


@router.message(Command("settings"))
async def cmd_settings(m: Message, lang: str):
    dpi = await storage.get_dpi(m.from_user.id)
    await m.answer(t(lang, "settings_title"), reply_markup=settings_menu(
        lambda k, **kw: t(lang, k, **kw), dpi))


# ---- admin ----

@router.message(Command("stats"))
async def cmd_stats(m: Message, lang: str):
    if m.from_user.id not in config.ADMIN_IDS:
        return await m.answer(t(lang, "admin_only"))
    s = await storage.stats_today()
    await m.answer(t(lang, "admin_stats", **s))


@router.message(Command("health"))
async def cmd_health(m: Message, lang: str):
    if m.from_user.id not in config.ADMIN_IDS:
        return await m.answer(t(lang, "admin_only"))
    n = len(list(config.TEMP_DIR.glob("**/*"))) if config.TEMP_DIR.exists() else 0
    await m.answer(t(lang, "admin_health", q=queue.position_estimate(), t=n))


@router.message(Command("broadcast"))
async def cmd_broadcast(m: Message, state: FSMContext, lang: str):
    if m.from_user.id not in config.ADMIN_IDS:
        return await m.answer(t(lang, "admin_only"))
    await state.set_state(Flow.waiting_broadcast)
    await m.answer(t(lang, "broadcast_prompt"))


@router.message(Flow.waiting_broadcast)
async def do_broadcast(m: Message, state: FSMContext, lang: str):
    if m.from_user.id not in config.ADMIN_IDS:
        return
    await state.clear()
    ok = 0
    for uid in await storage.all_user_ids():
        try:
            await m.copy_to(uid)
            ok += 1
        except Exception:
            pass
    await m.answer(t(lang, "broadcast_done", n=ok))

# ---------------- bot/handlers/photo.py ----------------

import io
import time
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (BufferedInputFile, InputMediaPhoto, Message)




EXT_OK = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def uid_dir(uid: int) -> Path:
    d = config.TEMP_DIR / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _download(bot: Bot, file_id: str, dest: Path) -> bool:
    f = await bot.get_file(file_id)
    if f.file_size and f.file_size > config.MAX_FILE_BYTES:
        return False
    await bot.download_file(f.file_path, dest)
    return True


@router.message(lambda m: m.photo or (m.document and m.document.mime_type
                and (m.document.mime_type.startswith("image/")
                     or m.document.mime_type == "application/octet-stream")))
async def on_photo(m: Message, bot: Bot, state: FSMContext, lang: str):
    uid = m.from_user.id
    lock = queue.user_lock(uid)
    if lock.locked():
        await m.answer(t(lang, "queue_position", n=1))
    async with lock:
        if m.document:
            ext = Path(m.document.file_name or "x.jpg").suffix.lower()
            if ext not in EXT_OK:
                return await m.answer(t(lang, "bad_file_type"))
        else:
            ext = ".jpg"
            await m.answer(t(lang, "send_file_hint"))

        progress = await m.answer(t(lang, "stage_checking"))
        d = uid_dir(uid)
        raw = d / f"orig{ext}"
        ok = await _download(bot, (m.document or m.photo[-1]).file_id, raw)
        if not ok:
            await progress.edit_text(t(lang, "file_too_large"))
            return

        sess = storage.default_session()
        sess["dpi"] = await storage.get_dpi(uid)
        # reuse preset if one exists
        preset = await storage.get_preset(uid)
        if preset:
            sess.update({k: v for k, v in preset.items() if k in sess})

        try:
            img = await queue.run_cpu(pipeline.load_photo, raw)
        except Exception:
            await progress.edit_text(t(lang, "bad_file_type"))
            return
        orig_path = d / "orig_rgb.png"
        work_path = d / "work.png"
        await queue.run_cpu(_save_variants, img, orig_path, work_path)
        sess["orig"], sess["work"] = str(orig_path), str(work_path)

        await progress.edit_text(t(lang, "stage_face"))
        work_img = await queue.run_cpu(lambda: __import__("PIL.Image", fromlist=["Image"]).open(work_path).convert("RGB"))
        faces = await queue.run_cpu(facesvc.detect_faces, work_img)

        if len(faces) > 1 and not m.media_group_id:
            sess["pending_faces"] = faces
            await storage.save_session(uid, sess)
            numbered = await queue.run_cpu(facesvc.draw_numbered, work_img, faces)
            buf = io.BytesIO(); numbered.save(buf, "PNG")
            await state.set_state(Flow.waiting_face_pick)
            await progress.delete()
            await m.answer_photo(BufferedInputFile(buf.getvalue(), "faces.png"),
                                 caption=t(lang, "multi_face", n=len(faces)),
                                 reply_markup=faces_kb(len(faces)))
            return

        await progress.edit_text(t(lang, "stage_bg"))
        t0 = time.time()
        sess = await queue.run_cpu(pipeline.auto_process, sess, d)
        await storage.save_session(uid, sess)
        await storage.bump_stat(jobs=1, ms=int((time.time() - t0) * 1000))

        if not sess.get("face"):
            await progress.delete()
            await m.answer(t(lang, "no_face"), reply_markup=noface_kb(
                lambda k, **kw: t(lang, k, **kw)))
            return

        await progress.edit_text(t(lang, "stage_finish"))
        await send_card(m, uid, lang, progress)


def _save_variants(img, orig_path: Path, work_path: Path):
    img.save(orig_path)
    pipeline.working_copy(img).save(work_path)


async def build_caption(uid: int, sess: dict, lang) -> str:
    tf = lambda k, **kw: t(lang, k, **kw)
    mm = layout.size_mm(sess)
    preview = await queue.run_cpu(pipeline.render_preview, sess)
    face = sess.get("face")
    if face and sess.get("work"):
        # scale face box to preview coords
        work = __import__("PIL.Image", fromlist=["Image"]).open(sess["work"])
        sx = preview.width / (work.width / 2)
        face_p = [int(v * sx / 2) for v in face]
    else:
        face_p = None
    orig_px = __import__("PIL.Image", fromlist=["Image"]).open(sess["orig"]).size
    checks = await queue.run_cpu(compliance.check, preview, face_p, mm,
                                 orig_px, sess.get("dpi") or 300)
    bg_name = sess.get("bg", "#FFFFFF")
    for k, v in BG_PRESETS.items():
        if v == bg_name:
            bg_name = tf(k)
            break
    if sess.get("transparent"):
        bg_name = tf("bg_transparent")
    title = tf("card_title", size_name=tf(f'size_{sess["size"]}')
               if sess["size"] != "custom" else "Custom",
               w=mm[0], h=mm[1], dpi=sess.get("dpi") or 300,
               bg=bg_name, glow=sess["recipe"].get("glow", 0),
               copies=sess["copies"], paper=sess["paper"].upper())
    return title + "\n" + compliance.render_text(checks, tf)


async def send_card(m_or_cb, uid: int, lang: str, progress=None):
    sess = await storage.get_session(uid)
    tf = lambda k, **kw: t(lang, k, **kw)
    preview = await queue.run_cpu(pipeline.render_preview, sess)
    buf = io.BytesIO(); preview.save(buf, "PNG")
    photo = BufferedInputFile(buf.getvalue(), "preview.png")
    caption = await build_caption(uid, sess, lang)
    kb = main_menu(tf)
    bot = (m_or_cb.message if hasattr(m_or_cb, "message") else m_or_cb).bot
    chat_id = uid
    card_id = sess.get("card_msg")
    if progress:
        try:
            await progress.delete()
        except Exception:
            pass
    if card_id:
        try:
            await bot.edit_message_media(
                InputMediaPhoto(media=photo, caption=caption, parse_mode="HTML"),
                chat_id=chat_id, message_id=card_id, reply_markup=kb)
            return
        except Exception:
            pass
    sent = await bot.send_photo(chat_id, photo, caption=caption,
                                parse_mode="HTML", reply_markup=kb)
    sess["card_msg"] = sent.message_id
    await storage.save_session(uid, sess)


@router.callback_query(Cb.filter(F.a == "face"))
async def pick_face(cb, state: FSMContext, lang: str):
    data = Cb.unpack(cb.data)
    uid = cb.from_user.id
    await cb.answer()
    sess = await storage.get_session(uid)
    if data.v == "manual":
        await state.set_state(Flow.waiting_manual_crop)
        sess["face"] = None
        sess = await queue.run_cpu(pipeline.auto_process, sess, uid_dir(uid))
        await storage.save_session(uid, sess)
        await send_card(cb, uid, lang)
        return
    idx = int(data.v)
    faces = sess.get("pending_faces") or []
    if idx < len(faces):
        sess["face"] = tuple(faces[idx])
    sess.pop("pending_faces", None)
    await state.clear()
    sess = await queue.run_cpu(pipeline.auto_process, sess, uid_dir(uid))
    await storage.save_session(uid, sess)
    await send_card(cb, uid, lang)

# ---------------- bot/handlers/menu.py ----------------

import io
import re
import shutil

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message





def tf_of(lang):
    return lambda k, **kw: t(lang, k, **kw)


async def _sess(uid):
    return await storage.get_session(uid)


async def _save_refresh(cb: CallbackQuery, sess: dict, lang: str):
    await storage.save_session(cb.from_user.id, sess)
    await send_card(cb, cb.from_user.id, lang)


def _push_history(sess: dict):
    sess["history"] = sess["history"][: sess["hpos"] + 1]
    sess["history"].append({"recipe": dict(sess["recipe"]),
                            "crop": dict(sess["crop"])})
    sess["history"] = sess["history"][-20:]
    sess["hpos"] = len(sess["history"]) - 1


@router.callback_query(Cb.filter(F.a == "noop"))
async def noop(cb: CallbackQuery):
    await cb.answer()


@router.callback_query(Cb.filter(F.a == "menu"))
async def open_menu(cb: CallbackQuery, lang: str):
    tf = tf_of(lang)
    sess = await _sess(cb.from_user.id)
    v = cb.data and Cb.unpack(cb.data).v
    await cb.answer()
    if v == "main":
        await send_card(cb, cb.from_user.id, lang)
    elif v == "size":
        await cb.message.edit_reply_markup(reply_markup=size_menu(tf, sess["size"]))
    elif v == "bg":
        await cb.message.edit_reply_markup(reply_markup=bg_menu(tf, sess.get("bg", "#FFFFFF")))
    elif v == "enh":
        await cb.message.edit_reply_markup(reply_markup=enhance_menu(tf, sess["recipe"]))
    elif v == "crop":
        mm = layout.size_mm(sess)
        await cb.message.edit_caption(
            caption=tf("crop_title", size=f"{mm[0]}×{mm[1]} mm"), parse_mode="HTML",
            reply_markup=crop_menu(tf, sess))
    elif v == "sheet":
        await sheet_preview(cb, sess, lang)
    elif v == "dl":
        await cb.message.edit_reply_markup(reply_markup=download_menu(tf))


@router.callback_query(Cb.filter(F.a == "size"))
async def pick_size(cb: CallbackQuery, state: FSMContext, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    if v == "custom":
        await state.set_state(Flow.waiting_custom_size)
        return await cb.message.answer(t(lang, "size_custom_prompt"))
    sess["size"] = v
    await _save_refresh(cb, sess, lang)


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|px)?\s*$", re.I)


@router.message(Flow.waiting_custom_size)
async def custom_size(m: Message, state: FSMContext, lang: str):
    g = _SIZE_RE.match(m.text or "")
    if not g:
        return await m.answer(t(lang, "size_bad"))
    w, h, unit = float(g[1]), float(g[2]), (g[3] or "mm").lower()
    if unit == "cm":
        w, h = w * 10, h * 10
    elif unit in ("in", "inch"):
        w, h = w * 25.4, h * 25.4
    elif unit == "px":
        w, h = w * 25.4 / 300, h * 25.4 / 300
    if not (5 <= w <= 300 and 5 <= h <= 400):
        return await m.answer(t(lang, "size_bad"))
    await state.clear()
    sess = await _sess(m.from_user.id)
    sess["size"], sess["custom_mm"] = "custom", [w, h]
    await storage.save_session(m.from_user.id, sess)
    await send_card(m, m.from_user.id, lang)


@router.callback_query(Cb.filter(F.a == "bg"))
async def pick_bg(cb: CallbackQuery, state: FSMContext, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    if v == "custom":
        await state.set_state(Flow.waiting_custom_colour)
        return await cb.message.answer(t(lang, "bg_custom_prompt"))
    if v == "transp":
        sess["transparent"] = True
    elif v == "reset":
        sess["bg"], sess["transparent"] = "#FFFFFF", False
        sess["edge"] = {"feather": 2, "shift": 0}
    else:
        sess["bg"] = bgsvc.BG_PRESETS[v]
        sess["transparent"] = False
    await _save_refresh(cb, sess, lang)


@router.message(Flow.waiting_custom_colour)
async def custom_colour(m: Message, state: FSMContext, lang: str):
    sess = await _sess(m.from_user.id)
    if m.photo:
        f = await m.bot.get_file(m.photo[-1].file_id)
        buf = await m.bot.download_file(f.file_path)
        from PIL import Image
        img = Image.open(buf)
        colour = bgsvc.average_colour(img)
    else:
        colour = (m.text or "").strip()
        if not bgsvc.valid_hex(colour):
            return await m.answer(t(lang, "bg_bad_hex"))
        if not colour.startswith("#"):
            colour = "#" + colour
        if len(colour) == 4:
            colour = "#" + "".join(c * 2 for c in colour[1:])
    await state.clear()
    sess["bg"], sess["transparent"] = colour.upper(), False
    await storage.save_session(m.from_user.id, sess)
    await send_card(m, m.from_user.id, lang)


@router.callback_query(Cb.filter(F.a.in_({"mask", "edge"})))
async def edge_tools(cb: CallbackQuery, lang: str):
    data = Cb.unpack(cb.data)
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    edge = sess.setdefault("edge", {"feather": 2, "shift": 0})
    if data.a == "mask":
        edge["shift"] = max(-10, min(10, edge["shift"] + (2 if data.v == "restore" else -2)))
    else:
        if data.v == "f+":
            edge["feather"] = min(10, edge["feather"] + 1)
        elif data.v == "f-":
            edge["feather"] = max(0, edge["feather"] - 1)
        elif data.v == "s+":
            edge["shift"] = min(10, edge["shift"] + 1)
        elif data.v == "s-":
            edge["shift"] = max(-10, edge["shift"] - 1)
    # re-refine mask from the raw cutout
    await queue.run_cpu(_re_refine, sess)
    await _save_refresh(cb, sess, lang)


def _re_refine(sess: dict):
    from PIL import Image
    cutout = Image.open(sess["cutout"])
    mask = cutout.getchannel("A") if cutout.mode == "RGBA" else Image.new("L", cutout.size, 255)
    e = sess["edge"]
    b.refine_mask(mask, e["feather"], e["shift"]).save(sess["mask"])


@router.callback_query(Cb.filter(F.a == "enh"))
async def enh_step(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    r = sess["recipe"]
    if v == "auto":
        r.update(await queue.run_cpu(_auto, sess))
    elif v == "boost":
        await queue.run_cpu(_boost, sess)
    elif v == "reset":
        r.update(storage.DEFAULT_RECIPE)
    else:
        key, sign = v[:-1], v[-1]
        cur = r.get(key, 0)
        r[key] = max(-10, min(10, cur + (1 if sign == "+" else -1)))
    _push_history(sess)
    await _save_refresh(cb, sess, lang)


def _auto(sess):
    from PIL import Image
    return ensvc.auto_recipe(Image.open(sess["aligned"]).convert("RGB"))


def _boost(sess):
    from PIL import Image
    boosted = ensvc.quality_boost(Image.open(sess["aligned"]).convert("RGB"))
    boosted.save(sess["aligned"])
    cut = Image.open(sess["cutout"]).convert("RGBA")
    cut2 = ensvc.quality_boost(cut.convert("RGB"))
    cut2.putalpha(cut.getchannel("A").resize(cut2.size))
    cut2.save(sess["cutout"])


@router.callback_query(Cb.filter(F.a == "look"))
async def look_pick(cb: CallbackQuery, lang: str):
    sess = await _sess(cb.from_user.id)
    sess["recipe"]["look"] = Cb.unpack(cb.data).v
    _push_history(sess)
    await cb.answer()
    await _save_refresh(cb, sess, lang)


@router.callback_query(Cb.filter(F.a == "hist"))
async def undo_redo(cb: CallbackQuery, lang: str):
    sess = await _sess(cb.from_user.id)
    v = Cb.unpack(cb.data).v
    h, p = sess["history"], sess["hpos"]
    await cb.answer()
    if v == "u" and p > 0:
        p -= 1
    elif v == "r" and p < len(h) - 1:
        p += 1
    sess["hpos"] = p
    snap = h[p]
    sess["recipe"], sess["crop"] = dict(snap["recipe"]), dict(snap["crop"])
    await _save_refresh(cb, sess, lang)


MOVE = {"normal": 1, "fine": 0.3, "coarse": 3}


@router.callback_query(Cb.filter(F.a.in_({"mv", "rot"})))
async def crop_move(cb: CallbackQuery, lang: str):
    data = Cb.unpack(cb.data)
    sess = await _sess(cb.from_user.id)
    c = sess["crop"]
    await cb.answer()
    step = MOVE.get(c.get("step", "normal"), 1)
    if data.a == "rot":
        c["rotate"] = (c.get("rotate", 0) + float(data.v)) % 360
    elif data.v == "l":
        c["dx"] = c.get("dx", 0) - step
    elif data.v == "r":
        c["dx"] = c.get("dx", 0) + step
    elif data.v == "u":
        c["dy"] = c.get("dy", 0) - step
    elif data.v == "d":
        c["dy"] = c.get("dy", 0) + step
    elif data.v == "z+":
        c["zoom"] = min(3.0, c.get("zoom", 1.0) + 0.05 * step)
    elif data.v == "z-":
        c["zoom"] = max(0.5, c.get("zoom", 1.0) - 0.05 * step)
    elif data.v == "flip":
        c["flip"] = not c.get("flip", False)
    elif data.v == "fit":
        c.update({"dx": 0, "dy": 0, "zoom": 1.0, "rotate": 0.0})
    elif data.v == "step":
        order = ["fine", "normal", "coarse"]
        c["step"] = order[(order.index(c.get("step", "normal")) + 1) % 3]
    elif data.v == "reset":
        c.update(storage.DEFAULT_CROP)
    _push_history(sess)
    await _save_refresh(cb, sess, lang)


@router.callback_query(Cb.filter(F.a == "cmp"))
async def compare(cb: CallbackQuery, lang: str):
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    img = await queue.run_cpu(pipeline.compare_image, sess)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)
    await cb.message.answer_photo(BufferedInputFile(buf.getvalue(), "compare.jpg"),
                                  caption=t(lang, "compare_caption"))


@router.callback_query(Cb.filter(F.a == "guides"))
async def guides(cb: CallbackQuery, lang: str):
    sess = await _sess(cb.from_user.id)
    sess["crop"]["guides"] = not sess["crop"].get("guides", False)
    await cb.answer()
    await _save_refresh(cb, sess, lang)


# ---------- sheet ----------

async def sheet_preview(cb_or_msg, sess: dict, lang: str):
    tf = tf_of(lang)
    uid = cb_or_msg.from_user.id
    final = await queue.run_cpu(pipeline.render_final, sess)
    preview_sheet, placed = await queue.run_cpu(
        lambda: layout.render_sheet(final, sess, preview=True))
    buf = io.BytesIO(); preview_sheet.save(buf, "PNG")
    fits = placed >= sess["copies"]
    kb = sheet_menu(tf, sess, fits)
    bot = cb_or_msg.bot if hasattr(cb_or_msg, "bot") else cb_or_msg.message.bot
    sheet_id = sess.get("sheet_msg")
    photo = BufferedInputFile(buf.getvalue(), "sheet.png")
    from aiogram.types import InputMediaPhoto
    caption = tf("sheet_title") if fits else tf(
        "copies_overflow", n=sess["copies"], paper=sess["paper"].upper(),
        max=layout.max_fit(layout.paper_mm(sess), layout.size_mm(sess),
                           sess["margin"], sess["gap"]))
    if sheet_id:
        try:
            await bot.edit_message_media(
                InputMediaPhoto(media=photo, caption=caption, parse_mode="HTML"),
                chat_id=uid, message_id=sheet_id, reply_markup=kb)
            return
        except Exception:
            pass
    sent = await bot.send_photo(uid, photo, caption=caption,
                                parse_mode="HTML", reply_markup=kb)
    sess["sheet_msg"] = sent.message_id
    await storage.save_session(uid, sess)


@router.callback_query(Cb.filter(F.a == "paper"))
async def paper(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    if v == "open":
        return await cb.message.edit_reply_markup(reply_markup=paper_menu(tf_of(lang)))
    sess["paper"] = v
    await storage.save_session(cb.from_user.id, sess)
    await sheet_preview(cb, sess, lang)


@router.callback_query(Cb.filter(F.a == "cp"))
async def copies(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    if v == "max":
        sess["copies"] = layout.max_fit(layout.paper_mm(sess),
                                        layout.size_mm(sess),
                                        sess["margin"], sess["gap"])
    elif v.startswith(("+", "-")):
        sess["copies"] = max(1, min(60, sess["copies"] + int(v)))
    else:
        sess["copies"] = max(1, min(60, int(v)))
    await storage.save_session(cb.from_user.id, sess)
    await sheet_preview(cb, sess, lang)


@router.callback_query(Cb.filter(F.a == "lay"))
async def lay_toggle(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    if v == "m":
        sess["margin"] = round((sess["margin"] + 2.5) % 15, 1)
    elif v == "g":
        sess["gap"] = round((sess["gap"] + 1) % 8, 1)
    elif v == "cuts":
        sess["cut_marks"] = not sess["cut_marks"]
    elif v == "border":
        sess["border"] = not sess["border"]
    await storage.save_session(cb.from_user.id, sess)
    await sheet_preview(cb, sess, lang)


# ---------- export ----------

@router.callback_query(Cb.filter(F.a == "dl"))
async def download(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    sess = await _sess(cb.from_user.id)
    uid = cb.from_user.id
    await cb.answer(t(lang, "stage_finish"))
    dpi = sess.get("dpi") or 300
    cache_key = f"{uid}:{v}:{sess['size']}:{sess['bg']}:{dpi}:{sess['copies']}:{sess['hpos']}"
    cached = await storage.cache_get(cache_key)
    try:
        if v == "pdf":
            if cached:
                await cb.message.answer_document(cached)
            else:
                final = await queue.run_cpu(pipeline.render_final, sess)
                pdf = await queue.run_cpu(pdf_export.sheet_pdf, final, sess)
                doc = BufferedInputFile(pdf, "passport_sheet.pdf")
                sent = await cb.message.answer_document(doc)
                await storage.cache_put(cache_key, sent.document.file_id)
        elif v in ("png", "jpg", "sheet"):
            final = await queue.run_cpu(pipeline.render_final, sess)
            sheet, _ = await queue.run_cpu(layout.render_sheet, final, sess, dpi)
            buf = io.BytesIO()
            sheet.save(buf, "PNG" if v != "jpg" else "JPEG", quality=95)
            name = f"passport_sheet.{ 'jpg' if v == 'jpg' else 'png'}"
            await cb.message.answer_document(BufferedInputFile(buf.getvalue(), name))
        elif v == "single":
            final = await queue.run_cpu(pipeline.render_final, sess)
            buf = io.BytesIO()
            if sess.get("transparent"):
                final_rgba = final.convert("RGBA")
                final_rgba.save(buf, "PNG")
                name = "passport_photo.png"
            else:
                final.save(buf, "JPEG", quality=95)
                name = "passport_photo.jpg"
            await cb.message.answer_document(BufferedInputFile(buf.getvalue(), name))
        await cb.message.answer(t(lang, "download_done"))
    except Exception:
        await cb.message.answer(t(lang, "error_generic"))


@router.callback_query(Cb.filter(F.a == "tkb"))
async def target_kb(cb: CallbackQuery, lang: str):
    target = int(Cb.unpack(cb.data).v)
    sess = await _sess(cb.from_user.id)
    await cb.answer()
    final = await queue.run_cpu(pipeline.render_final, sess)
    data, kb = await queue.run_cpu(pdf_export.to_jpeg_target_kb, final, target)
    await cb.message.answer_document(
        BufferedInputFile(data, f"photo_{kb}kb.jpg"),
        caption=t(lang, "target_kb_done", kb=kb))


# ---------- settings callbacks ----------

@router.callback_query(Cb.filter(F.a == "dpi"))
async def dpi_toggle(cb: CallbackQuery, lang: str):
    uid = cb.from_user.id
    new = 600 if await storage.get_dpi(uid) == 300 else 300
    await storage.set_dpi(uid, new)
    sess = await _sess(uid)
    sess["dpi"] = new
    await storage.save_session(uid, sess)
    await cb.answer()
    await cb.message.edit_reply_markup(reply_markup=settings_menu(tf_of(lang), new))


@router.callback_query(Cb.filter(F.a == "preset"))
async def preset(cb: CallbackQuery, lang: str):
    v = Cb.unpack(cb.data).v
    uid = cb.from_user.id
    sess = await _sess(uid)
    if v == "save":
        await storage.save_preset(uid, {
            "size": sess["size"], "custom_mm": sess["custom_mm"],
            "bg": sess["bg"], "transparent": sess["transparent"],
            "recipe": sess["recipe"]})
        await cb.answer(t(lang, "preset_saved"))
    else:
        p = await storage.get_preset(uid)
        if not p:
            return await cb.answer(t(lang, "no_preset"), show_alert=True)
        sess.update(p)
        await cb.answer(t(lang, "preset_applied"))
        await _save_refresh(cb, sess, lang)


@router.callback_query(Cb.filter(F.a == "deldata"))
async def deldata(cb: CallbackQuery, state: FSMContext, lang: str):
    if Cb.unpack(cb.data).v == "ask":
        await cb.answer(t(lang, "delete_data_confirm"), show_alert=True)
        from aiogram.utils.keyboard import InlineKeyboardBuilder
        b = InlineKeyboardBuilder()
        b.row(btn(t(lang, "set_delete_data"), "deldata", "yes"))
        return await cb.message.edit_reply_markup(reply_markup=b.as_markup())
    uid = cb.from_user.id
    await state.clear()
    d = config.TEMP_DIR / str(uid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    await storage.clear_session(uid)
    await storage.delete_user_data(uid)
    await cb.answer()
    await cb.message.answer(t(lang, "data_deleted"))


@router.callback_query(Cb.filter(F.a == "retry"))
async def retry(cb: CallbackQuery, lang: str):
    await cb.answer()
    await send_card(cb, cb.from_user.id, lang)


# shortcuts as commands


@router.message(Command("size"))
async def c_size(m: Message, lang: str):
    sess = await _sess(m.from_user.id)
    await m.answer(t(lang, "size_title"),
                   reply_markup=size_menu(tf_of(lang), sess["size"]))


@router.message(Command("background"))
async def c_bg(m: Message, lang: str):
    sess = await _sess(m.from_user.id)
    await m.answer(t(lang, "bg_title"),
                   reply_markup=bg_menu(tf_of(lang), sess.get("bg", "#FFFFFF")))


@router.message(Command("copies"))
async def c_copies(m: Message, state: FSMContext, lang: str):
    await state.set_state(Flow.waiting_copies)
    await m.answer(t(lang, "sheet_type_number"))


@router.message(Flow.waiting_copies)
async def c_copies_num(m: Message, state: FSMContext, lang: str):
    if not (m.text or "").isdigit():
        return await m.answer(t(lang, "sheet_type_number"))
    await state.clear()
    sess = await _sess(m.from_user.id)
    sess["copies"] = max(1, min(60, int(m.text)))
    await storage.save_session(m.from_user.id, sess)
    await sheet_preview(m, sess, lang)


@router.message(Command("paper"))
async def c_paper(m: Message, lang: str):
    await m.answer(t(lang, "sheet_paper", paper=""), reply_markup=paper_menu(tf_of(lang)))

# ============================== Single shared router ==============================

# ============================== Startup and lifecycle ==============================
def setup_logging() -> None:
    config.ensure_dirs()
    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL)
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = RotatingFileHandler(config.DATA_DIR / "bot.log", maxBytes=2_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(sh)
        root.addHandler(fh)

async def set_commands(bot: Bot) -> None:
    commands = [
        ("start", "Start"), ("help", "How it works"), ("settings", "Settings"),
        ("size", "Photo size"), ("background", "Background colour"),
        ("copies", "Copies on sheet"), ("paper", "Paper size"),
        ("language", "Language"), ("reset", "Reset edits"),
        ("cancel", "Clear session"), ("privacy", "Privacy"),
    ]
    en = [BotCommand(command=c, description=d) for c, d in commands]
    hi_desc = ["शुरू करें", "मदद", "सेटिंग्स", "फोटो साइज़", "बैकग्राउंड रंग",
               "शीट पर कॉपी", "पेपर साइज़", "भाषा", "रीसेट", "सत्र साफ़ करें", "निजता"]
    hi = [BotCommand(command=c, description=d) for (c, _), d in zip(commands, hi_desc)]
    await bot.set_my_commands(en)
    await bot.set_my_commands(hi, language_code="hi")

async def cleanup_loop() -> None:
    ttl = config.TEMP_TTL_MINUTES * 60
    while True:
        await asyncio.sleep(300)
        now = time.time()
        if not config.TEMP_DIR.exists():
            continue
        for directory in config.TEMP_DIR.iterdir():
            try:
                if directory.is_dir():
                    latest = max((f.stat().st_mtime for f in directory.glob("**/*")
                                  if f.is_file()), default=directory.stat().st_mtime)
                    if now - latest > ttl:
                        shutil.rmtree(directory, ignore_errors=True)
            except OSError:
                logging.getLogger("bot").debug("Could not clean %s", directory, exc_info=True)

async def on_startup(bot: Bot) -> None:
    config.ensure_dirs()
    await storage.init()
    load_locales()
    await set_commands(bot)
    asyncio.create_task(cleanup_loop(), name="temp-cleanup")

async def on_shutdown(bot: Bot) -> None:
    queue.shutdown()
    await storage.close()

def build() -> tuple[Bot, Dispatcher]:
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set. Configure it as a Railway environment variable.")
    bot = Bot(config.BOT_TOKEN,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.update.middleware(ErrorMiddleware())
    dp.message.middleware(FloodMiddleware())
    dp.callback_query.middleware(FloodMiddleware())
    dp.message.middleware(UserMiddleware())
    dp.callback_query.middleware(UserMiddleware())
    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    return bot, dp

def main() -> None:
    setup_logging()
    bot, dp = build()
    if config.MODE == "webhook":
        if not config.WEBHOOK_URL:
            raise SystemExit("WEBHOOK_URL is required when MODE=webhook")
        from aiohttp import web
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
        app = web.Application()
        SimpleRequestHandler(dp, bot).register(app, path="/webhook")
        setup_application(app, dp, bot=bot)

        async def register_webhook(_app):
            await bot.set_webhook(config.WEBHOOK_URL, drop_pending_updates=True)

        app.on_startup.append(register_webhook)
        web.run_app(app, host=config.WEBHOOK_HOST, port=config.WEBHOOK_PORT)
    else:
        asyncio.run(dp.start_polling(bot, drop_pending_updates=True))

if __name__ == "__main__":
    main()
