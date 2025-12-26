from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
import uuid
from datetime import datetime
from bson.objectid import ObjectId
import bcrypt

from preprocessing import preprocess_image
from models import UserCreate, UserUpdate, UserResponse
from pymongo import MongoClient
import os
import base64


# =========================
# MongoDB Configuration
# =========================


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://RohitBarshile:Rohit%40123@cluster0.elfzhpa.mongodb.net/"
)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["alzheimers_db"]
predictions_collection = db["predictions"]
users_collection = db["users"]


# =========================
# Helper Functions
# =========================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hashed password"""
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


# =========================
# Model & Class Names
# =========================

# {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

model = tf.keras.models.load_model("my_model.h5")
LOADED_MODEL_FILENAME = "my_model.h5"


# =========================
# FastAPI App Initialization
# =========================

app = FastAPI(
    title="Alzheimer MRI Classification API",
    description="Predict Alzheimer stage from MRI images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# User CRUD Endpoints
# =========================

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """Create a new user"""
    # Check if user with email already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user document
    user_doc = {
        "user_id": str(uuid.uuid4()),
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "password_hash": hash_password(user.password),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Insert into MongoDB
    result = users_collection.insert_one(user_doc)
    
    return {
        "user_id": user_doc["user_id"],
        "email": user_doc["email"],
        "full_name": user_doc["full_name"],
        "role": user_doc["role"],
        "created_at": user_doc["created_at"],
        "updated_at": user_doc["updated_at"]
    }


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by ID"""
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "role": user["role"],
        "created_at": user["created_at"],
        "updated_at": user["updated_at"]
    }


@app.get("/users", response_model=list[UserResponse])
async def list_users():
    """List all users"""
    users = list(users_collection.find({}, {"password_hash": 0}))
    
    return [
        {
            "user_id": user["user_id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "created_at": user["created_at"],
            "updated_at": user["updated_at"]
        }
        for user in users
    ]


@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_update: UserUpdate):
    """Update user information"""
    # Find user
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if email is being changed and if it already exists
    if user_update.email and user_update.email != user["email"]:
        existing_user = users_collection.find_one({"email": user_update.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Prepare update data
    update_data = {}
    if user_update.email:
        update_data["email"] = user_update.email
    if user_update.full_name:
        update_data["full_name"] = user_update.full_name
    if user_update.role:
        update_data["role"] = user_update.role
    
    update_data["updated_at"] = datetime.utcnow()
    
    # Update user
    users_collection.update_one({"user_id": user_id}, {"$set": update_data})
    
    # Get updated user
    updated_user = users_collection.find_one({"user_id": user_id})
    
    return {
        "user_id": updated_user["user_id"],
        "email": updated_user["email"],
        "full_name": updated_user["full_name"],
        "role": updated_user["role"],
        "created_at": updated_user["created_at"],
        "updated_at": updated_user["updated_at"]
    }


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    """Delete a user"""
    # Find and delete user
    result = users_collection.delete_one({"user_id": user_id})
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return None


# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # measure preprocessing + prediction time
        start_time = datetime.utcnow()
        img = preprocess_image(image_bytes)

        preds = model.predict(img)   # shape (1, 4)
        probs = preds[0]
        end_time = datetime.utcnow()

        processing_ms = (end_time - start_time).total_seconds() * 1000.0

        class_probabilities = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        # determine top prediction
        top_index = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[top_index]
        confidence = float(np.max(probs))

        # Save to DB with additional metadata to allow analytics
        prediction_id = str(uuid.uuid4())
        predictions_collection.insert_one({
            "prediction_id": prediction_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "image": image_bytes,
            "class_probabilities": class_probabilities,
            "predicted_class": predicted_label,
            "confidence": round(confidence, 4),
            "processing_ms": round(processing_ms, 2),
            "created_at": datetime.utcnow()
        })

        # Return probability mapping (same as before)
        return class_probabilities

    except Exception as e:
        return {"error": str(e)}


@app.get("/analytics")
async def analytics():
    """Aggregate basic analytics from the predictions collection.

    Returns: {
        total_predictions: int,
        avg_confidence: float | None,   # percentage 0-100
        active_users: int | None,       # count of distinct user_id if present
        avg_processing_ms: float | None
    }
    """
    try:
        total = predictions_collection.count_documents({})

        # compute average confidence: prefer stored 'confidence' field if present
        pipeline_conf = [
            {"$match": {"confidence": {"$exists": True}}},
            {"$group": {"_id": None, "avgConf": {"$avg": "$confidence"}}}
        ]
        conf_cursor = list(predictions_collection.aggregate(pipeline_conf))
        if conf_cursor and conf_cursor[0].get("avgConf") is not None:
            avg_confidence = float(conf_cursor[0]["avgConf"]) * 100.0 if conf_cursor[0]["avgConf"] <= 1 else float(conf_cursor[0]["avgConf"])  # handle 0-1 or 0-100
        else:
            # fallback: compute from class_probabilities max value
            pipeline_fallback = [
                {"$project": {"maxProb": {"$max": {"$map": {"input": {"$objectToArray": "$class_probabilities"}, "as": "cp", "in": "$$cp.v"}}}}},
                {"$group": {"_id": None, "avgMax": {"$avg": "$maxProb"}}}
            ]
            fb = list(predictions_collection.aggregate(pipeline_fallback))
            if fb and fb[0].get("avgMax") is not None:
                avg_confidence = float(fb[0]["avgMax"]) * 100.0 if fb[0]["avgMax"] <= 1 else float(fb[0]["avgMax"])
            else:
                avg_confidence = None

        # active users: count distinct user_id if it's present in documents
        try:
            distinct_users = predictions_collection.distinct("user_id")
            active_users = len(distinct_users) if distinct_users is not None else None
            if active_users == 0:
                # no user_id field present — return None to indicate unavailable
                active_users = None
        except Exception:
            active_users = None

        # average processing time
        pipeline_time = [
            {"$match": {"processing_ms": {"$exists": True}}},
            {"$group": {"_id": None, "avgMs": {"$avg": "$processing_ms"}}}
        ]
        tcur = list(predictions_collection.aggregate(pipeline_time))
        avg_processing_ms = float(tcur[0]["avgMs"]) if tcur and tcur[0].get("avgMs") is not None else None

        return {
            "total_predictions": total,
            "avg_confidence": round(avg_confidence, 2) if avg_confidence is not None else None,
            "active_users": active_users,
            "avg_processing_ms": round(avg_processing_ms, 2) if avg_processing_ms is not None else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
async def list_predictions(limit: int = 100):
    """Return recent prediction records (most recent first).

    Each record includes: prediction_id, filename, predicted_class, confidence (0-100),
    class_probabilities (as mapping with numeric probabilities), processing_ms,
    created_at, and image as a base64 data URL (`data:{content_type};base64,...`).
    """
    try:
        cursor = predictions_collection.find().sort("created_at", -1).limit(limit)
        results = []
        for doc in cursor:
            # prepare image as base64 data URL if available
            image_data = None
            if doc.get("image"):
                try:
                    b = doc["image"]
                    # `b` may be bytes or pymongo Binary
                    if isinstance(b, (bytes, bytearray)):
                        img_bytes = bytes(b)
                    else:
                        img_bytes = bytes(b)
                    content_type = doc.get("content_type", "image/png")
                    b64 = base64.b64encode(img_bytes).decode('ascii')
                    image_data = f"data:{content_type};base64,{b64}"
                except Exception:
                    image_data = None

            class_probs = doc.get("class_probabilities") or {}
            # normalize confidence to percentage (if stored 0-1)
            confidence = doc.get("confidence")
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    if conf_val <= 1:
                        confidence_pct = round(conf_val * 100.0, 2)
                    else:
                        confidence_pct = round(conf_val, 2)
                except Exception:
                    confidence_pct = None
            else:
                # fallback: take max of class_probabilities
                try:
                    maxp = max([float(v) for v in (class_probs.values() or [0])])
                    confidence_pct = round((maxp * 100.0) if maxp <= 1 else maxp, 2)
                except Exception:
                    confidence_pct = None

            # convert class_probabilities values to percentages for frontend convenience
            class_probs_pct = {k: (float(v) * 100.0 if float(v) <= 1 else float(v)) for k, v in class_probs.items()}

            results.append({
                "prediction_id": doc.get("prediction_id") or str(doc.get("_id")),
                "filename": doc.get("filename"),
                "predicted_class": doc.get("predicted_class"),
                "confidence": confidence_pct,
                "class_probabilities": class_probs_pct,
                "processing_ms": doc.get("processing_ms"),
                "created_at": doc.get("created_at"),
                "image_data": image_data,
            })

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """Return available model files in the backend folder (.h5) with basic metadata."""
    try:
        base_dir = os.getcwd()
        model_files = []
        for fname in os.listdir(base_dir):
            if fname.lower().endswith('.h5') or fname.lower().endswith('.hdf5'):
                path = os.path.join(base_dir, fname)
                try:
                    st = os.stat(path)
                    model_files.append({
                        'filename': fname,
                        'path': path,
                        'size_bytes': st.st_size,
                        'modified_at': datetime.utcfromtimestamp(st.st_mtime),
                        'active': (fname == LOADED_MODEL_FILENAME)
                    })
                except Exception:
                    # skip files we can't stat
                    continue

        # sort newest first
        model_files.sort(key=lambda x: x['modified_at'], reverse=True)

        # convert datetimes to isoformat for JSON
        for m in model_files:
            m['modified_at'] = m['modified_at'].isoformat() + 'Z'

        return model_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

