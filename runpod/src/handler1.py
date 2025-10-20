# handler.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import torch
import runpod
from dotenv import load_dotenv
from RS_utils1 import ThreeHeadCSACActor, encode_state_features, format_recommendation, get_or_refresh_connection, custom_serializer, extract_responses, convert_height_to_inches, format_skus, format_boosts
import json
from datetime import datetime
import traceback
import pandas as pd
import requests
from pydantic_settings import BaseSettings
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("/app/logs/application.log", maxBytes=5*1024*1024, backupCount=7),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

actor = None
variant_ids = None
categories = None
redis_client = None
boost_ids = None
sku_ids = None
sku_dict = None
boost_dict = None

# Load env for Redis
import redis
redis_client = redis.StrictRedis(host=os.getenv('REDIS_HOST', '127.0.0.1'), port=6379, db=0)

def get_from_cache(response_id):
    try:
        cached_data = redis_client.get(response_id)
        if cached_data:
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None
    

def store_in_cache(response_id, data):
    try:
        redis_client.set(response_id, json.dumps(data, default=custom_serializer))
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")

try:
    conn = get_or_refresh_connection()
    cursor = conn.cursor()
    cursor.execute("""    
        SELECT DISTINCT 
            br.VARIANTID,
            vn.CATEGORY
        FROM INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BASE_RESULT br
        INNER JOIN INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_NUTRITION_DATA vn
            ON CAST(br.VARIANTID AS STRING) = CAST(vn.VARIANTID AS STRING)
        WHERE br.VARIANTID IS NOT NULL
        ORDER BY br.VARIANTID
    """)
    variant_df = cursor.fetch_pandas_all()
    variant_ids = [str(v) for v in variant_df['VARIANTID'].tolist()]
    categories = [str(c) for c in variant_df['CATEGORY'].tolist()]
    logger.info("Variant data loaded successfully")

    # Load unique boost items
    cursor.execute("""
        SELECT DISTINCT ITEMID, ITEMNAME, HANDLE
        FROM INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BOOST
        WHERE ITEMID IS NOT NULL
        ORDER BY ITEMID
    """)
    boost_df = cursor.fetch_pandas_all()
    # boost_ids = [str(i) for i in boost_df['ITEMID'].tolist()]
    boost_dict = {str(row['ITEMID']): {'item_name': str(row['ITEMNAME']), 'handle': str(row.get('HANDLE', ''))} for row in boost_df.to_dict('records')}
    boost_ids = list(boost_dict.keys())
    logger.info(f"Loaded {len(boost_ids)} unique boost items")
    
    # Load unique SKU items
    cursor.execute("""
        WITH RankedItems AS (
            SELECT 
                VARIANTID, 
                ITEMNAME,
                ITEMPRICE,
                ROW_NUMBER() OVER (PARTITION BY VARIANTID ORDER BY ITEMNAME) AS rn
            FROM INGEST_DATA.TRUENUTRITION_BUILD.ORDERS_LINESUMMARY
            WHERE VARIANTID IN (
                SELECT DISTINCT VARIANTID
                FROM INGEST_DATA.TRUENUTRITION_BUILD.COACH_SKU_RESULT
            )
        ),
        RankedVariantInfo AS (
            SELECT 
                VARIANTID,
                SHOPIFY_PRODUCT_ID, 
                SHOPIFY_VARIANT_ID,
                SHORT_DESCRIPTION,
                IMAGE_URL,
                HANDLE,
                ROW_NUMBER() OVER (PARTITION BY VARIANTID ORDER BY SHOPIFY_VARIANT_ID) AS rn
            FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANTINFORMATION
        ),
        RankedOrderLines AS (
            SELECT
                VARIANTID,
                ITEMPRICE,
                ITEMTYPE,
                SKU,
                ROW_NUMBER() OVER (PARTITION BY VARIANTID ORDER BY ITEMPRICE) AS rn
            FROM INGEST_DATA.TRUENUTRITION_BUILD.ORDERS_LINESUMMARY
        )
        SELECT  
            vi.SHOPIFY_PRODUCT_ID, 
            vi.SHOPIFY_VARIANT_ID,
            vi.SHORT_DESCRIPTION,
            vi.IMAGE_URL,
            MAX(rol.ITEMPRICE) AS ITEMPRICE,  -- Aggregate ITEMPRICE to get one value per VARIANTID
            vi.HANDLE,
            rol.ITEMTYPE,
            rol.SKU,
            ri.VARIANTID,
            ri.ITEMNAME
        FROM RankedItems ri
        LEFT JOIN RankedVariantInfo vi
            ON ri.VARIANTID = vi.VARIANTID AND vi.rn = 1  -- Ensure only one row per VARIANTID from VARIANTINFORMATION
        LEFT JOIN RankedOrderLines rol
            ON ri.VARIANTID = rol.VARIANTID AND rol.rn = 1  -- Ensure only the first row from ORDERS_LINESUMMARY
        WHERE ri.rn = 1  -- Ensure only the first row per VARIANTID from ORDERS_LINESUMMARY
        GROUP BY
            vi.SHOPIFY_PRODUCT_ID, 
            vi.SHOPIFY_VARIANT_ID,
            vi.SHORT_DESCRIPTION,
            vi.IMAGE_URL,
            vi.HANDLE,
            rol.ITEMTYPE,
            rol.SKU,
            ri.VARIANTID,
            ri.ITEMNAME
        ORDER BY ri.VARIANTID;
    """)
    sku_df = cursor.fetch_pandas_all()
    sku_data = sku_df.to_dict('records')  # List of dicts, one per unique VARIANTID
    sku_dict = {str(row['VARIANTID']): row for row in sku_data}
    sku_ids = list(sku_dict.keys())  # Unique variant IDs as strings
    logger.info(f"Loaded {len(sku_ids)} unique SKU items with details")
    
    state_dim = 120 
    base_dim = len(variant_ids)
    boost_dim = len(boost_ids)
    sku_dim = len(sku_ids)
    actor = ThreeHeadCSACActor(state_dim=state_dim, base_dim=base_dim, boost_dim=boost_dim, sku_dim=sku_dim)
    with open('/app/models/improved_csac_actor.pth', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        actor.load_state_dict(state_dict)
    actor.eval()
    logger.info(f"Actor model loaded successfully with state_dim={state_dim}, base_dim={base_dim}, boost_dim={boost_dim}, sku_dim={sku_dim}")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
    raise Exception(f"Failed to initialize: {str(e)}")

AUTHORIZATION_TOKEN = settings.authorization_token
FORM_ID = settings.form_id
TYPEFORM_API_TOKEN = settings.typeform_api_token

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     if credentials.credentials != AUTHORIZATION_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing token",
#         )
    

def handler(job):
    try:
        start_time = time.time()
        job_input = job.get("input", {})
        # email = job_input.get("email")
        response_id = job_input.get("response_id")
        if not response_id:
            logger.error("Missing response_id in input")
            return {"error": "Missing response_id"}
        
        logger.info(f"Fetching recommendations for response_id: {response_id}")

        # Check cache first
        recommendations = get_from_cache(response_id)
        if recommendations:
            logger.info(f"Cache hit for response_id: {response_id}")
            total_time = time.time() - start_time
            recommendations["total_time_taken"] = total_time
            return recommendations
        

        url = f"https://api.typeform.com/forms/{FORM_ID}/responses?included_response_ids={response_id}"
        headers = {
            "Authorization": f"Bearer {TYPEFORM_API_TOKEN}",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        typeform_data = response.json()

        if "items" not in typeform_data or not typeform_data["items"]:
            return {"error": "No responses found in Typeform"}
            
        typeform_item = typeform_data["items"][0]
        survey_date = typeform_item['submitted_at']
        user_name = None
        user_email = None
        for answer in typeform_item['answers']:
            field_ref = answer['field']['ref']
            if field_ref == settings.field_ref_firstname:
                user_name = answer.get('text', None)
            if field_ref == settings.field_ref_email:
                user_email = answer.get('email', None)

        user_df = extract_responses(typeform_item)
        state = encode_state_features(user_df)

        if state is None:
            return {"error": "Failed to encode state features"}
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get three actions from model
        base_action, boost_action, sku_action = actor(state_tensor)
        base_action = base_action.detach().numpy()[0]
        boost_action = boost_action.detach().numpy()[0]
        sku_action = sku_action.detach().numpy()[0]
        
        # Convert to lists of floats
        base_action = [float(x) for x in base_action]
        boost_action = [float(x) for x in boost_action]
        sku_action = [float(x) for x in sku_action]
        
        # Format each recommendation
        base_recommendation = format_recommendation(base_action, variant_ids, categories)
        boost_recommendation = format_boosts(boost_action, boost_ids, boost_dict)
        sku_recommendation = format_skus(sku_action, sku_ids, sku_dict)

        # DEBUG: Log raw model outputs
        logger.info(f"STATE VECTOR: {state[:10]}... (first 10 values)")
        logger.info(f"STATE SHAPE: {state.shape}")
        logger.info(f"BASE ACTION - min: {base_action.min():.6f}, max: {base_action.max():.6f}, mean: {base_action.mean():.6f}")
        logger.info(f"BASE ACTION (first 5): {base_action[:5]}")
        logger.info(f"BOOST ACTION - min: {boost_action.min():.6f}, max: {boost_action.max():.6f}, mean: {boost_action.mean():.6f}")
        logger.info(f"BOOST ACTION (first 5): {boost_action[:5]}")
        logger.info(f"SKU ACTION - min: {sku_action.min():.6f}, max: {sku_action.max():.6f}, mean: {sku_action.mean():.6f}")
        logger.info(f"SKU ACTION (first 5): {sku_action[:5]}")

        # Also log the intermediate features
        logger.info(f"USER PROFILE - GENDER: {user_df.iloc[0].get('GENDER')}, AGE: {user_df.iloc[0].get('AGE')}, WEIGHT: {user_df.iloc[0].get('WEIGHT')}")

        # Extract profile from the processed DataFrame
        profile_data = user_df.iloc[0]  # Get first row
        profile_list = [
            {
                "gender": profile_data.get("GENDER"),
                # "age": profile_data.get("AGE"),
                # "height_inches": profile_data.get("HEIGHT"),
                # "weight": profile_data.get("WEIGHT"),
                "age": int(profile_data.get("AGE")) if pd.notna(profile_data.get("AGE")) else None,
                "height_inches": int(profile_data.get("HEIGHT")) if pd.notna(profile_data.get("HEIGHT")) else None,
                "weight": float(profile_data.get("WEIGHT")) if pd.notna(profile_data.get("WEIGHT")) else None,
                "body_type": profile_data.get("BODYTYPE"),
                "body_fat": profile_data.get("BODYFAT"),
                "diet": profile_data.get("DIET"),
                "diet_restrictions": profile_data.get("DIETRESTRICTIONS"),
                "macros": profile_data.get("MACROS"),
                "goal_primary": profile_data.get("PRIMARYGOAL"),
                "goal_secondary": profile_data.get("SECONDARYGOAL"),
                "activity": profile_data.get("ACTIVITY"),
                "activity_level": profile_data.get("ACTIVITYLEVEL"),
                "mix_intended_use": profile_data.get("INTENDEDUSE"),
                "mix_frequency": profile_data.get("FREQUENCY"),
                "mix_timing": profile_data.get("MIXTIMING"),
                "health_issues": profile_data.get("HEALTHCONCERNS"),
            }
        ]

        recommendations_dict = {
            "email": user_email,
            "survey_date": survey_date,
            "first_name": user_name,
            "profile": profile_list,
            "recommended_protein_mix": base_recommendation["protein_mix"],
            "recommended_boosts": boost_recommendation["boosts"],
            "recommended_skus": sku_recommendation["sku_recommendations"]
        }

        recommendations_json = json.dumps(recommendations_dict, default=custom_serializer)
        
        # Store in cache
        store_in_cache(response_id, recommendations_dict)

        # Snowflake logging
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.LOG_COACH_RS
            (ResponseId, ResponseJSON, Email, StatusCode, StatusMessage)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (response_id, recommendations_json, user_email, 200, "success")
        )
        
        cursor.execute("SELECT MAX(ID) FROM INGEST_DATA.TRUENUTRITION_BUILD.LOG_COACH_RS")
        log_id = cursor.fetchone()[0]
        
        for item in base_recommendation['protein_mix']:
            cursor.execute(
                """
                INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BASE_RESULT
                (LOG_ID, RESPONSEID, CATEGORY, ITEMNAME, VARIANTID, PERCENTOFMIX)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (log_id, response_id, item['category'], item['item_name'], item['variant_id'], item['percent'])
            )

        # Log boosts 
        for item in boost_recommendation['boosts']:
            cursor.execute(
                """
                INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BOOST
                (LOG_ID, RESPONSEID, ITEMNAME, ITEMID, HANDLE)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (log_id, response_id, item['item_name'], item['item_id'], item['handle'])
            )
        
        # Log SKUs
        for item in sku_recommendation['sku_recommendations']:
            variant_item = item['variants']['items'][0]
            cursor.execute(
                """
                INSERT INTO INGEST_DATA.TRUENUTRITION_BUILD.COACH_SKU_RESULT
                (LOG_ID, RESPONSEID, VARIANTID, SKU)
                VALUES (%s, %s, %s, %s)
                """,
                # (log_id, response_id, item['sku_id'], item['sku'])variant_id
                (log_id, response_id, item['variant_id'], variant_item['sku'])
            )

        conn.commit()
        
        total_time = time.time() - start_time
        return {
            "response_id": response_id,
            "email": user_email,
            "survey_date": survey_date,
            "first_name": user_name,
            "recommendations": recommendations_dict,
            # "timestamp": datetime.utcnow().isoformat(),
            "total_time_taken": total_time
        }
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Failed to generate recommendations: {str(e)}"}

runpod.serverless.start({"handler": handler})

# job_input = {
#     "input": {
#         # "email": "firedudepete@gmail.com",
#         "response_id": "t8rmpia65cg3ywqpbl5vlut8rmpiaepk"
#     }
# }

# if __name__ == "__main__":
#     output = handler(job_input)
#     print("Handler Output:")
#     print(output)