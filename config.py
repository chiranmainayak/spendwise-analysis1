import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "student_expenditure_40_months.xlsx"
DB_FILE = BACKEND_DIR / "spendwise.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE.as_posix()}")

MONTHLY_ALLOWANCE = 5000.0
CATEGORIES = [
    "Clothes",
    "Education",
    "Entertainment",
    "Food",
    "Health",
    "Recharge",
    "Transport"
]

RETRAIN_INTERVAL = 25
HELD_OUT_SHEET = "Apr_2029"
