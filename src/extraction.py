import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.logger import get_logger

load_dotenv()
logger = get_logger()

RAW_DATA_PATH = "data/raw/obt_raw.csv"
OBT_TABLE = os.getenv("OBT_TABLE", "ml_schema.obt_annonce")

def get_engine():
    try:
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "avito_db")
        
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        return create_engine(conn_str)
    except Exception as e:
        logger.error(f"[extraction] Failed to initialize database engine: {e}")
        raise

def extract_obt(save_csv: bool = True) -> pd.DataFrame:
    logger.info("[extraction] Connecting to fetch fresh OBT data...")
    try:
        engine = get_engine()
        query = f"SELECT * FROM {OBT_TABLE};"
        df = pd.read_sql(query, engine)
        logger.info(f"[extraction] Extracted {len(df):,} rows successfully from database.")
    except Exception as e:
        logger.warning(f"[extraction] Database connection offline ({e}). checking for local uploaded files.")
        if os.path.exists("obt_raw.csv"):
            df = pd.read_csv("obt_raw.csv")
            logger.info(f"[extraction] Loaded {len(df):,} rows from uploaded obt_raw.csv file.")
        elif os.path.exists(RAW_DATA_PATH):
            df = pd.read_csv(RAW_DATA_PATH)
            logger.info(f"[extraction] Loaded {len(df):,} rows from local fallback: {RAW_DATA_PATH}")
        else:
            raise FileNotFoundError("Could not locate any database tables or local raw CSV snapshots.")
            
    if save_csv:
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"[extraction] Snapshot saved to {RAW_DATA_PATH}")
        
    return df