# rs_utils.py
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import snowflake.connector
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import holidays
import json
import time
from config import settings
import traceback
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

logger = logging.getLogger(__name__)

class ThreeHeadCSACActor(nn.Module):
    def __init__(self, state_dim, base_dim, boost_dim, sku_dim, max_action=1.0, hidden_dim=256, sparsity_weight=0.01):
        super(ThreeHeadCSACActor, self).__init__()
        self.base_dim = base_dim
        self.boost_dim = boost_dim
        self.sku_dim = sku_dim
        self.max_action = max_action
        self.log_std_min = -10
        self.log_std_max = 2
        self.sparsity_weight = sparsity_weight

        # Shared base network
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )

        # Base head
        self.base_head = nn.Linear(hidden_dim, base_dim * 2)
        # Boost head
        self.boost_head = nn.Linear(hidden_dim, boost_dim * 2)
        # SKU head
        self.sku_head = nn.Linear(hidden_dim, sku_dim * 2)

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        base_features = self.base_network(state)

        # Base action
        base_mu_logstd = self.base_head(base_features)
        base_mu, base_log_std = base_mu_logstd[:, :self.base_dim], base_mu_logstd[:, self.base_dim:]
        base_log_std = torch.clamp(base_log_std, self.log_std_min, self.log_std_max)
        base_std = base_log_std.exp()
        base_dist = torch.distributions.Normal(base_mu, base_std)
        base_action_raw = base_dist.rsample()
        base_action_squashed = torch.tanh(base_action_raw)
        base_action_normalized = torch.softmax(base_action_squashed * 5.0, dim=-1) * self.max_action

        # Boost action
        boost_mu_logstd = self.boost_head(base_features)
        boost_mu, boost_log_std = boost_mu_logstd[:, :self.boost_dim], boost_mu_logstd[:, self.boost_dim:]
        boost_log_std = torch.clamp(boost_log_std, self.log_std_min, self.log_std_max)
        boost_std = boost_log_std.exp()
        boost_dist = torch.distributions.Normal(boost_mu, boost_std)
        boost_action_raw = boost_dist.rsample()
        boost_action_squashed = torch.tanh(boost_action_raw)
        boost_action_normalized = torch.sigmoid(boost_action_squashed) * self.max_action

        # SKU action
        sku_mu_logstd = self.sku_head(base_features)
        sku_mu, sku_log_std = sku_mu_logstd[:, :self.sku_dim], sku_mu_logstd[:, self.sku_dim:]
        sku_log_std = torch.clamp(sku_log_std, self.log_std_min, self.log_std_max)
        sku_std = sku_log_std.exp()
        sku_dist = torch.distributions.Normal(sku_mu, sku_std)
        sku_action_raw = sku_dist.rsample()
        sku_action_squashed = torch.tanh(sku_action_raw)
        sku_action_normalized = torch.sigmoid(sku_action_squashed) * self.max_action

        return base_action_normalized, boost_action_normalized, sku_action_normalized


def get_snowflake_connection(max_retries=3, retry_delay=30):
    for attempt in range(1, max_retries + 1):
        try:
            # Load RSA private key directly from environment variable
            private_key_str = os.getenv("rsa_key_coach_snow")
            if not private_key_str:
                raise ValueError(" rsa_key_coach_snow not found in environment variables.")

            # Convert string with "\n" into proper PEM format
            private_key_bytes = private_key_str.encode().replace(b"\\n", b"\n")

            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=os.getenv("SNOWFLAKE_SSH_PASS").encode(),
                backend=default_backend()
            )

            private_key_der = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            logger.info(f"Attempt {attempt} to connect to Snowflake...")
            conn = snowflake.connector.connect(
                user=os.getenv("SNOWFLAKE_USER"),
                private_key=private_key_der,
                account=os.getenv("SNOWFLAKE_ACCOUNT"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA"),
            )
            logger.info("Snowflake connection established successfully.")
            return conn

        except snowflake.connector.errors.DatabaseError as e:
            err_msg = str(e)
            logger.warning(f" Snowflake connection failed (attempt {attempt}): {err_msg}")
            if "JWT token is invalid" in err_msg or "Failed to authenticate" in err_msg:
                if attempt < max_retries:
                    logger.info(f" Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries exceeded. Could not connect to Snowflake.")
                    raise
            else:
                raise

conn = None


def get_or_refresh_connection():
    global conn
    if conn is None:
        conn = get_snowflake_connection()
    else:
        try:
            conn.cursor().execute("SELECT 1")
        except:
            logger.info("Connection lost, refreshing...")
            conn = get_snowflake_connection()
    return conn


def convert_height_to_inches(height_str):
    try:
        if height_str == "Under 5 feet":
            height_str = "4'11\""
        if height_str == "Over 7 feet":
            height_str = "7'1\""
        feet, inches = height_str.split("'")
        inches = inches.replace('"', '').strip()
        return int(feet) * 12 + int(inches)
    except (ValueError, AttributeError):
        return None
    except Exception as e:
        logger.error(f"Error converting height to inches: {e}")
        return None
    
    

from sklearn.preprocessing import OneHotEncoder  

# Define the predefined categorical options globally (add this at the module level)
CATEGORICAL_OPTIONS = {
    'GENDER': ['Male', 'Female', 'Other'],
    'BODYTYPE': ['Skinny/Lean', 'Athletic/Fit', 'Muscular/Bodybuilder', 'Average/ Normal', 'Curvy', 'Overweight', 'Obese'],
    'BODYFAT': ['Lean/low fat', 'Normal / moderate amount of body fat', 'High/a lot of body fat', 'Obese/very high amount of body fat'],
    'DIET': ['Animal based (some plants)', 'Carnivore', 'Omnivore/Other', 'Pescatarian', 'Plant based (some meat)', 'Vegan', 'Vegetarian'],
    'DIETRESTRICTIONS': ['Gluten intolerance or sensitivity', 'Lactose intolerance', 'Lactose sensitivity(some dairy)', 'Nut allergy', 'Egg allergy', 'Seafood allergy', 'Other allergies/sensitivities', 'No soy', 'No Restrictions'],
    'MACROS': ['High protein', 'moderate protein', 'low protein', 'High carb', 'moderate carb', 'moderate carbs', 'low carb', 'High fat', 'moderate fat', 'low fat', 'carbs and fat split evenly', 'protein'],
    'PRIMARYGOAL': ['Improve overall health', 'Build muscle', 'Lose fat', 'Improve endurance', 'Increase energy', 'General Maintenance'],
    'SECONDARYGOAL': ['Improve overall health', 'Build muscle', 'Lose fat', 'Improve endurance', 'Increase energy', 'General Maintenance', 'Unknown'],
    'ACTIVITYLEVEL': ['Sedentary', 'Lightly active', 'Moderately active', 'Very active (intense exercise or sports 6-7 days a week)', 'Super active'],
    'ACTIVITY': ['Cycling', 'General cardio', 'HIIT', 'MMA/Combat Sports', 'Other Sports', 'Powerlifting', 'Recreational Sports','Resistance training', 'Rowing', 'Running', 'Spinning', 'Strength Training', 'Swimming', 'Walking/ Hiking', 'Yoga/ Pilates'],
    'INTENDEDUSE': ['Meal / Snack replacement', 'Pre-workout', 'Intra-workout', 'Post-workout / Recovery', 'Hitting calorie and macro goals'],
    'FREQUENCY': ['Once daily', 'Once every other day', 'Once Weekly', 'Multiple times daily'],
    'MIXTIMING': ['Morning/ First meal', 'Morning snack', 'Lunchtime', 'Afternoon snack', 'Evening/ dinner meal', 'Late night snack'],
    'HEALTHCONCERNS': ['Dehydration', 'Joint Pain', 'Libido', 'Sleep', 'Immune system', 'Digestion /Gut Health', 'Skin health', 'Need more fiber', 'Cognition/ Brain function', 'Chronic Stress', 'Hormone imbalance', 'Trouble gaining muscle', 'Trouble losing fat', 'Low energy', 'Vitamin/ Mineral deficiency', 'Other']
}

# The extract_responses function remains unchanged
def extract_responses(typeform_item):
    response_dict = {
        "EMAIL": None,
        "GENDER": None,
        "AGE": None,
        "HEIGHT": None,
        "WEIGHT": None,
        "BODYTYPE": None,
        "BODYFAT": None,
        "DIET": None,
        "DIETRESTRICTIONS": None,
        "MACROS": None,
        "PRIMARYGOAL": None,
        "SECONDARYGOAL": None,
        "ACTIVITY": None,
        "ACTIVITYLEVEL": None,
        "INTENDEDUSE": None,
        "FREQUENCY": None,
        "MIXTIMING": None,
        "HEALTHCONCERNS": None,
        "CREATEDON": pd.to_datetime(typeform_item['submitted_at'])
    }

    for answer in typeform_item['answers']:
        field_ref = answer['field']['ref']
        if field_ref == settings.field_ref_email:
            response_dict['EMAIL'] = answer.get('email', None)
        elif field_ref == settings.field_ref_gender:
            response_dict['GENDER'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_age:
            response_dict['AGE'] = answer.get('number', None)
        elif field_ref == settings.field_ref_height_inches:
            height_str = answer.get('text', None)
            response_dict['HEIGHT'] = convert_height_to_inches(height_str)
        elif field_ref == settings.field_ref_weight:
            response_dict['WEIGHT'] = answer.get('number', None)
        elif field_ref == settings.field_ref_body_type:
            response_dict['BODYTYPE'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_body_fat:
            response_dict['BODYFAT'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_diet:
            response_dict['DIET'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_diet_restrictions:
            response_dict['DIETRESTRICTIONS'] = ', '.join(answer.get('choices', {}).get('labels', []))
        elif field_ref == settings.field_ref_macros:
            response_dict['MACROS'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_goal_primary:
            response_dict['PRIMARYGOAL'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_goal_secondary:
            response_dict['SECONDARYGOAL'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_activity:
            response_dict['ACTIVITY'] = ', '.join(answer.get('choices', {}).get('labels', []))
        elif field_ref == settings.field_ref_activity_level:
            response_dict['ACTIVITYLEVEL'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_mix_intended_use:
            response_dict['INTENDEDUSE'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_mix_frequency:
            response_dict['FREQUENCY'] = answer.get('choice', {}).get('label', None)
        elif field_ref == settings.field_ref_mix_timing:
            response_dict['MIXTIMING'] = ', '.join(answer.get('choices', {}).get('labels', []))
        elif field_ref == settings.field_ref_health_issues:
            response_dict['HEALTHCONCERNS'] = ', '.join(answer.get('choices', {}).get('labels', []))

    current_time = datetime.now()
    response_dict['TNLINEITEMID'] = 0.0
    response_dict['MIN PURCHASE TIMESTAMP'] = current_time

    user_response_df = pd.DataFrame([response_dict])
    return user_response_df


# compute_dynamic_features remains unchanged
def compute_dynamic_features(timestamp):
    us_holidays = holidays.US()
    time_encoder = OneHotEncoder(sparse_output=False)
    day_encoder = OneHotEncoder(sparse_output=False)
    
    time_categories = np.array([['Morning'], ['Afternoon'], ['Evening']])
    day_categories = np.array([['Monday'], ['Tuesday'], ['Wednesday'], ['Thursday'], ['Friday'], ['Saturday'], ['Sunday']])
    time_encoder.fit(time_categories)
    day_encoder.fit(day_categories)

    hour = timestamp.hour
    time_of_day = 'Morning' if 8 <= hour < 12 else 'Afternoon' if 12 <= hour < 16 else 'Evening'
    day_of_week = timestamp.strftime('%A')
    is_holiday = 1.0 if timestamp.date() in us_holidays else 0.0

    time_encoded = time_encoder.transform([[time_of_day]])[0]
    day_encoded = day_encoder.transform([[day_of_week]])[0]
    dynamic_features = np.concatenate([[is_holiday], time_encoded, day_encoded])
    return dynamic_features.astype(np.float32)

# Modified encode_state_features for fixed one-hot/multi-hot encoding
def encode_state_features(user_df):
    try:    
        expected_numerical_cols = ['WEIGHT', 'HEIGHT', 'AGE']
        numerical_cols = [col for col in expected_numerical_cols if col in user_df.columns]
        if not numerical_cols:
            logger.warning("No expected numerical columns found in profile data")
            numerical_encoded = np.array([])
        else:
            numerical_data = user_df[numerical_cols].copy()

            def convert_height(height):
                if pd.isna(height) or height == 'Unknown':
                    return np.nan
                if height == 'under 5 feet':
                    return 59
                try:
                    height = str(height).replace('"', '')
                    if "'" in height:
                        feet, inches = height.split("'")
                        return int(feet) * 12 + int(inches)
                    else:
                        return float(height)
                except:
                    logger.warning(f"Invalid height format: {height}")
                    return np.nan
                
            if 'HEIGHT' in numerical_data.columns:
                numerical_data['HEIGHT'] = numerical_data['HEIGHT'].apply(convert_height)
            numerical_data.fillna(numerical_data.mean(), inplace=True)
            numerical_min = numerical_data.min()
            numerical_max = numerical_data.max()
            denominator = numerical_max - numerical_min + 1e-6
            numerical_encoded = (numerical_data - numerical_min) / denominator

        categorical_cols = [
            'GENDER', 'BODYTYPE', 'BODYFAT', 'DIET', 'DIETRESTRICTIONS',
            'MACROS', 'PRIMARYGOAL', 'SECONDARYGOAL', 'ACTIVITYLEVEL',
            'ACTIVITY', 'INTENDEDUSE', 'FREQUENCY', 'MIXTIMING', 'HEALTHCONCERNS'
        ]
        categorical_dfs = []
        available_categorical_cols = [col for col in categorical_cols if col in user_df.columns]
        
        for col in available_categorical_cols:
            user_value = user_df[col].iloc[0]  # Single row
            if pd.isna(user_value):
                selected = set()
            else:
                # Split by comma for multi-select, strip whitespace
                selected = set([v.strip() for v in str(user_value).split(',') if v.strip()])
            
            # Use predefined options for this column
            if col in CATEGORICAL_OPTIONS:
                options = CATEGORICAL_OPTIONS[col]
                # Create one-hot/multi-hot: 1 if option in selected, else 0
                dct = {f"{col}_{opt}": [1 if opt in selected else 0] for opt in options}
                categorical_dfs.append(pd.DataFrame(dct, index=user_df.index))
                # Optional: Log for debugging (comment out if too verbose)
                # logger.info(f"Column {col} selected: {sorted(selected)} out of {len(options)} options")
            else:
                logger.warning(f"No predefined options for column {col}")
        
        categorical_combined = pd.concat(categorical_dfs, axis=1) if categorical_dfs else pd.DataFrame(index=user_df.index)
        

        # Compute dynamic features using CREATEDON timestamp
        createdon_col = 'CREATEDON'
        if createdon_col in user_df.columns and not pd.isna(user_df[createdon_col].iloc[0]):
            createdon = pd.to_datetime(user_df[createdon_col].iloc[0])
        else:
            createdon = datetime.now()
            logger.warning(f"No valid CREATEDON found; using current time: {createdon}")
        dynamic_part = compute_dynamic_features(createdon)
        logger.info(f"Dynamic features length: {len(dynamic_part)}")
        numerical_part = numerical_encoded.values[0] if len(numerical_encoded) > 0 else np.array([])
        categorical_part = categorical_combined.values[0] if len(categorical_combined) > 0 else np.array([])
        
        logger.info(f"Numerical features: {len(numerical_part)}, Categorical features: {len(categorical_part)}, Dynamic features: {len(dynamic_part)}")
        state_parts = [part for part in [numerical_part, categorical_part, dynamic_part] if len(part) > 0]
        if state_parts:
            state = np.concatenate(state_parts, dtype=np.float32)
        else:
            logger.error("No valid state features could be extracted")
            return None
        expected_dim = 120
        if len(state) != expected_dim:
            logger.warning(f"State dimension is {len(state)}, expected {expected_dim}. Padding/truncating...")
            if len(state) < expected_dim:
                state = np.pad(state, (0, expected_dim - len(state)), mode='constant')
            else:
                state = state[:expected_dim]
        return state

    except Exception as e:
        logger.error(f"State encoding error: {str(e)}\n{traceback.format_exc()}")
        return None
    

def format_recommendation(action, variant_ids, categories):
    protein_mix = []
    total_percent = sum(action)
    if total_percent > 0:
        base_action_normalized = [float(a) / total_percent for a in action]
    else:
        base_action_normalized = [float(a) for a in action]
    for i, (variant_id, category) in enumerate(zip(variant_ids, categories)):
        item_name = get_item_name(variant_id)
        if base_action_normalized[i] > 0.05:
            protein_mix.append({
                "variant_id": str(variant_id),
                "percent": float(base_action_normalized[i] * 100),
                "category": str(category),
                "item_name": str(item_name)
            })
    return {"protein_mix": protein_mix}


def format_boosts(boost_action, boost_ids, boost_dict):
    boosts = []
    for i, boost_id in enumerate(boost_ids):
        if boost_action[i] > 0.5:  
            item_name = get_boost_name(boost_id)
            row = boost_dict.get(str(boost_id))  # Use the dict
            if not row:
                continue
            boosts.append({
                "item_id": str(boost_id),
                "weight": float(boost_action[i]),  
                "item_name": item_name,
                "handle": row['handle']
            })
    return {"boosts": boosts}


# def get_sku_name(variant_id):
#     try:
#         conn = get_or_refresh_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             WITH RankedItems AS (
#                 SELECT 
#                     VARIANTID, 
#                     ITEMNAME,
#                     ROW_NUMBER() OVER (PARTITION BY VARIANTID ORDER BY ITEMNAME) AS rn
#                 FROM INGEST_DATA.TRUENUTRITION_BUILD.ORDERITEMSBYEMAIL
#                 WHERE VARIANTID = %s
#             )
#             SELECT ITEMNAME
#             FROM RankedItems
#             WHERE rn = 1
#         """, (variant_id,))
#         result = cursor.fetchone()
#         return str(result[0]) if result else "unknown"
#     except Exception as e:
#         logger.error(f"Error fetching SKU name for variant_id {variant_id}: {str(e)}")
#         return "unknown"

def format_skus(sku_action, sku_ids, sku_dict):
    skus = []
    for i, variant_id in enumerate(sku_ids):
        if sku_action[i] > 0.5:  
            row = sku_dict.get(variant_id)
            if not row:
                logger.warning(f"No SKU data found for variant_id: {variant_id}")
                continue
            # product_url = f"https://truenutrition.com/products/{row['HANDLE']}?response_id={response_id}"
            variant_item = {
                "shopify_variant_id": row['SHOPIFY_VARIANT_ID'],
                "name": row['ITEMNAME'],
                "price": float(row['ITEMPRICE']),
                "sku": row['SKU']
            }
            rec = {
                "variant_id": str(row['VARIANTID']),
                "name": row['ITEMNAME'],
                "shopify_product_id": row['SHOPIFY_PRODUCT_ID'],
                "image_url": row['IMAGE_URL'],
                "handle": row['HANDLE'],
                # "product_url": product_url,
                "short_description": row['SHORT_DESCRIPTION'],
                "variants": {
                    "type": row.get('ITEMTYPE', 'Flavor'),
                    "items": [variant_item]
                }
            }
            skus.append(rec)
    return {"sku_recommendations": skus}


def get_item_name(variant_id):
    try:
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT PRODUCTNAME FROM INGEST_DATA.TRUENUTRITION_BUILD.VARIANT_CLASSIFICATION WHERE VARIANTID = %s",
            (variant_id,)
        )
        result = cursor.fetchone()
        item_name = result[0] if result else "unknown"
        return str(item_name)
    except Exception as e:
        logger.error(f"Error fetching item name for variant_id {variant_id}: {str(e)}")
        return "unknown"
    
def get_boost_name(item_id):
    try:
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ITEMNAME FROM INGEST_DATA.TRUENUTRITION_BUILD.COACH_RECOMMENDED_BOOST WHERE ITEMID = %s",
            (item_id,)
        )
        result = cursor.fetchone()
        return str(result[0]) if result else "unknown"
    except Exception as e:
        logger.error(f"Error fetching boost name for item_id {item_id}: {str(e)}")
        return "unknown"
    
def get_sku_name(variant_id):
    try:
        conn = get_or_refresh_connection()
        cursor = conn.cursor()
        cursor.execute("""
            WITH RankedItems AS (
                SELECT 
                    VARIANTID, 
                    ITEMNAME,
                    ROW_NUMBER() OVER (PARTITION BY VARIANTID ORDER BY ITEMNAME) AS rn
                FROM INGEST_DATA.TRUENUTRITION_BUILD.ORDERITEMSBYEMAIL
                WHERE VARIANTID = %s
            )
            SELECT ITEMNAME
            FROM RankedItems
            WHERE rn = 1
        """, (variant_id,))
        result = cursor.fetchone()
        return str(result[0]) if result else "unknown"
    except Exception as e:
        logger.error(f"Error fetching SKU name for variant_id {variant_id}: {str(e)}")
        return "unknown" 

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)