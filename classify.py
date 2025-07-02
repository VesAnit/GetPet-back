from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.security import HTTPBearer
from google.cloud import vision
from google.cloud import storage
import os
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import logging
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import requests
import uuid
from sqlalchemy import and_

from models import Pet, Announcement, User as UserModel, Favorite
from schemas import PetCreate, AnnouncementCreate, SearchRequest, AnnouncementResponse, ValidateImageResponse, FavoriteCreate, FavoriteResponse
from database import get_db
from auth import get_current_user_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
router = APIRouter(tags=["classify"])

MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
GCS_BUCKET_NAME = "pet-backend-uploads"
GCS_UPLOAD_DIR = "uploads"

def get_credentials():
    """Get credentials for Google Cloud."""
    try:
        if "CLOUD_RUN" in os.environ:

            from google.auth import default
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            return credentials
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
        return service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        logger.error(f"Error getting credentials: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

try:
    credentials = get_credentials()
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
except Exception as e:
    logger.error(f"Error initializing Google Cloud clients: {e}")
    raise HTTPException(status_code=500, detail="Failed to connect to Google Cloud services")

try:
    with open("breeds_map.json", "r", encoding="utf-8") as f:
        BREEDS_MAP = json.load(f)
except Exception as e:
    logger.error(f"Error loading breeds_map.json: {e}")
    BREEDS_MAP = {"types": {"Dog": "Dog", "Cat": "Cat"}, "dog": {}, "cat": {}}
    raise HTTPException(status_code=500, detail="Error loading breeds list")

security = HTTPBearer()

def check_image_for_animals(image_data: bytes) -> bool:
    """Check that exactly one animal (cat or dog) is in the image."""
    try:
        image = vision.Image(content=image_data)
        objects = vision_client.object_localization(image=image).localized_object_annotations
        logger.info(f"Objects found by Vision API: {[obj.name for obj in objects]}")

        animal_count = sum(1 for obj in objects if obj.name.lower() in ["dog", "cat"])
        invalid_objects = [
            obj.name for obj in objects
            if obj.name.lower() in ["person", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "rabbit"]
        ]

        if invalid_objects:
            logger.warning(f"Invalid objects found: {invalid_objects}")
            return False
        if animal_count != 1:
            logger.warning(f"Found {animal_count} animals. Exactly 1 cat or dog is required.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking image: {e}")
        raise HTTPException(status_code=400, detail=f"Image validation error: {str(e)}")

def detect_breed(image_data: bytes) -> Dict[str, Optional[str]]:
    """Detect animal type and breed using Vision API."""
    try:
        image = vision.Image(content=image_data)
        response = vision_client.web_detection(image=image)
        logger.info("Response from Web Detection API received.")

        if response.error.message:
            logger.error(f"Vision API error: {response.error.message}")
            return {"type": None, "breed": None, "confidence": 0}

        web_entities = response.web_detection.web_entities
        if not web_entities:
            logger.info("Web Detection did not return web entities.")
            return {"type": None, "breed": None, "confidence": 0}

        animal_type = None
        breed = None
        confidence = 0
        plausible_breeds = []

        for entity in web_entities:
            desc_lower = entity.description.lower().replace(" ", "_")
            if desc_lower in BREEDS_MAP["dog"]:
                plausible_breeds.append((entity.score, desc_lower, "Dog"))
            elif desc_lower in BREEDS_MAP["cat"]:
                plausible_breeds.append((entity.score, desc_lower, "Cat"))

        if plausible_breeds:
            plausible_breeds.sort(key=lambda x: x[0], reverse=True)
            confidence, breed, animal_type = plausible_breeds[0]
            logger.info(f"Best breed: {breed} (Type: {animal_type}, Confidence: {confidence:.2f})")

            if confidence < 0.4:
                breed = "mixed_breed"
                logger.info("Confidence below 0.4 threshold, setting to Mixed Breed")
        else:
            logger.warning("Failed to determine breed or animal type")

        return {"type": animal_type, "breed": breed, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error detecting breed: {e}")
        return {"type": None, "breed": None, "confidence": 0}

async def process_images_for_animal_type(
    image_data_pairs: List[Tuple[UploadFile, bytes]], for_creation: bool = False
) -> Tuple[Optional[str], List[str], Optional[str]]:
    """Process images to determine animal type and breed."""
    if len(image_data_pairs) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed")

    suggested_animal_type = None
    suggested_breeds = []
    max_confidence_breed = None
    max_confidence = 0.0

    for idx, (image, image_data) in enumerate(image_data_pairs):
        logger.info(f"Processing image {image.filename}, size: {len(image_data)} bytes")
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail={"message": "Empty image", "invalid_image_index": idx}
            )

        ext = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={"message": f"Unsupported format: {ext}", "invalid_image_index": idx}
            )

        if not check_image_for_animals(image_data):
            raise HTTPException(
                status_code=400,
                detail={"message": "Exactly one animal must be in the photo", "invalid_image_index": idx}
            )

        detection_result = detect_breed(image_data)
        if not detection_result["type"]:
            raise HTTPException(
                status_code=400,
                detail={"message": "Failed to determine animal type", "invalid_image_index": idx}
            )

        animal_type = detection_result["type"].lower()
        if suggested_animal_type is None:
            suggested_animal_type = animal_type
        elif animal_type != suggested_animal_type:
            raise HTTPException(
                status_code=400,
                detail={"message": "All images must contain the same animal type", "invalid_image_index": idx}
            )

        if detection_result["breed"]:
            breed_dict = BREEDS_MAP["dog"] if detection_result["type"] == "Dog" else BREEDS_MAP["cat"]
            breed = breed_dict.get(detection_result["breed"], detection_result["breed"])
            if breed not in suggested_breeds:
                suggested_breeds.append(breed)
            if for_creation and detection_result["confidence"] > max_confidence:
                max_confidence = detection_result["confidence"]
                max_confidence_breed = breed

    return suggested_animal_type, suggested_breeds, max_confidence_breed

def check_text_for_toxicity(text: str) -> bool:
    """Check text for toxicity using Mistral AI."""
    if not MISTRAL_TOKEN:
        logger.warning("Skipping text check: MISTRAL_TOKEN is missing")
        return False

    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_TOKEN}", "Content-Type": "application/json"}
    prompt = (
        f"You are a highly strict content moderator tasked with classifying text as 'offensive' or 'non-offensive'. "
        f"Classify the provided text with maximum sensitivity, marking any content that could be interpreted as harmful, toxic, or inappropriate as 'offensive'. "
        f"Offensive text includes, but is not limited to: "
        f"- Profanity, vulgar language, or obscene expressions (e.g., swear words, crude slang). "
        f"- Insults, personal attacks, or derogatory remarks targeting individuals or groups. "
        f"- Threats, including explicit, implied, or veiled threats of physical, emotional, or psychological harm (e.g., 'kill her', 'she doesn't deserve to live', 'you'll regret this'). "
        f"- Discrimination or hate speech based on race, ethnicity, gender, sexual orientation, religion, nationality, age, disability, or any other characteristic. "
        f"- Toxic behavior, such as shaming, bullying, gaslighting, humiliation, or passive-aggressive remarks (e.g., 'you're worthless', 'nobody cares about you'). "
        f"- Calls to violence, incitement to harm, or promotion of extremist ideologies. "
        f"- Sexual content, innuendos, or suggestive language, including objectification or harassment. "
        f"- Propaganda, misinformation, or content promoting hatred, division, or discrimination. "
        f"- Subtle or sarcastic remarks that could be interpreted as harmful or manipulative in context (e.g., 'maybe you should just disappear'). "
        f"- Any text that could cause emotional distress, fear, or discomfort, even if not explicitly aggressive. "
        f"If the text is in a language other than English, translate it to English first and then classify it. "
        f"Consider the tone, intent, and context: even neutral words can be offensive if used maliciously. "
        f"Examples: "
        f"- Offensive: 'Kill her now', 'You're a pathetic loser', 'Go back to your country', 'She deserves to suffer', 'You're too fat to be here'. "
        f"- Non-offensive: 'I like your shirt', 'Can you help me?', 'The weather is nice today'. "
        f"Text to classify: '{text}' "
        f"Response: Return only 'offensive' or 'non-offensive'."
    )
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "system", "content": "You are a highly strict content moderator with zero tolerance for toxic or harmful content."}, {"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 50
    }

    try:
        response = requests.post(mistral_url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        classification = response.json()["choices"][0]["message"]["content"].strip()
        is_toxic = classification == "offensive"
        logger.info(f"Mistral classification: {classification}")
        return is_toxic
    except Exception as e:
        logger.error(f"Error checking text: {e}")
        return False

def pet_form(
    animal_type: str = Form(..., description="Animal type (cat or dog)"),
    name: Optional[str] = Form(None, description="Animal name"),
    gender: str = Form(..., description="Gender (M or F)"),
    age: Optional[int] = Form(None, description="Animal age"),
    breed: Optional[str] = Form(None, description="Animal breed"),
    color: Optional[str] = Form(None, description="Animal color"),
) -> PetCreate:
    return PetCreate(
        animal_type=animal_type,
        name=name,
        gender=gender,
        age=age,
        breed=breed,
        color=color
    )

def announcement_form(
    keywords: Optional[str] = Form(None, description="Keywords"),
    description: Optional[str] = Form(None, description="Announcement description"),
    location: Optional[str] = Form(None, description="Animal location")
) -> AnnouncementCreate:
    return AnnouncementCreate(
        keywords=keywords,
        description=description,
        location=location
    )

async def upload_to_gcs(image_data: bytes, file_name: str) -> str:
    """Upload file to Google Cloud Storage and return its URL."""
    try:
        blob = bucket.blob(f"{GCS_UPLOAD_DIR}/{file_name}")
        blob.upload_from_string(image_data, content_type=f"image/{file_name.split('.')[-1].lower()}")
        url = blob.public_url
        logger.info(f"Uploaded file to GCS: {url}")
        return url
    except Exception as e:
        logger.error(f"Error uploading file to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file to GCS: {str(e)}")

async def delete_from_gcs(file_url: str):
    """Delete file from Google Cloud Storage."""
    try:
        file_name = file_url.split(f"/{GCS_UPLOAD_DIR}/")[-1]
        blob = bucket.blob(f"{GCS_UPLOAD_DIR}/{file_name}")
        if blob.exists():
            blob.delete()
            logger.info(f"Deleted file from GCS: {file_url}")
    except Exception as e:
        logger.error(f"Error deleting file from GCS: {e}")

@router.post("/validate_images", response_model=ValidateImageResponse)
async def validate_images(
    image: UploadFile = File(...),
    previous_type: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Validate image for announcement creation."""
    logger.info(f"Validating image for user_id: {current_user_id}")
    try:
        image_data = await image.read()
        if not image_data:
            return ValidateImageResponse(
                suggested_animal_type=None,
                max_confidence_breed=None,
                confidence=0,
                error_message="Empty image",
                invalid_image_index=0
            )

        ext = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return ValidateImageResponse(
                suggested_animal_type=None,
                max_confidence_breed=None,
                confidence=0,
                error_message=f"Unsupported format: {ext}",
                invalid_image_index=0
            )

        if not check_image_for_animals(image_data):
            return ValidateImageResponse(
                suggested_animal_type=None,
                max_confidence_breed=None,
                confidence=0,
                error_message="Exactly one animal must be in the photo",
                invalid_image_index=0
            )

        detection_result = detect_breed(image_data)
        if not detection_result["type"]:
            return ValidateImageResponse(
                suggested_animal_type=None,
                max_confidence_breed=None,
                confidence=0,
                error_message="Failed to determine animal type",
                invalid_image_index=0
            )

        animal_type = detection_result["type"].lower()
        if previous_type and animal_type != previous_type.lower():
            return ValidateImageResponse(
                suggested_animal_type=animal_type,
                max_confidence_breed=None,
                confidence=0,
                error_message=f"Animal type does not match previous: {previous_type}",
                invalid_image_index=0
            )

        max_confidence_breed = None
        if detection_result["breed"]:
            breed_dict = BREEDS_MAP["dog"] if detection_result["type"] == "Dog" else BREEDS_MAP["cat"]
            max_confidence_breed = breed_dict.get(detection_result["breed"], detection_result["breed"])

        return ValidateImageResponse(
            suggested_animal_type=animal_type,
            max_confidence_breed=max_confidence_breed,
            confidence=detection_result["confidence"],
            error_message=None,
            invalid_image_index=None
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/create_announcement", response_model=AnnouncementResponse)
async def create_announcement(
    pet: PetCreate = Depends(pet_form),
    announcement: AnnouncementCreate = Depends(announcement_form),
    images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Create a new announcement."""
    logger.info(f"Creating announcement for user_id: {current_user_id}")
    user = db.query(UserModel).filter(UserModel.id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    image_urls = []
    try:
        image_data_pairs = []
        for image in images:
            ext = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
            image_data = await image.read()
            if not image_data:
                raise HTTPException(status_code=400, detail=f"Empty image: {image.filename}")
            image_data_pairs.append((image, image_data))

        suggested_animal_type, suggested_breeds, max_confidence_breed = await process_images_for_animal_type(
            image_data_pairs, for_creation=True
        )
        if suggested_animal_type != pet.animal_type.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Specified animal type '{pet.animal_type}' does not match detected '{suggested_animal_type}'"
            )

        for image, image_data in image_data_pairs:
            ext = image.filename.split('.')[-1].lower() if '.' in image.filename else 'jpg'
            file_name = f"{uuid.uuid4()}.{ext}"
            file_url = await upload_to_gcs(image_data, file_name)
            image_urls.append(file_url)

        full_text = " ".join(filter(None, [
            pet.name, pet.breed, pet.color, str(pet.age) if pet.age is not None else "",
            announcement.description, announcement.keywords, announcement.location or ""
        ]))
        if full_text and check_text_for_toxicity(full_text):
            raise HTTPException(status_code=400, detail="Announcement rejected due to policy violation")

        new_pet = Pet(
            animal_type=pet.animal_type,
            name=pet.name,
            gender=pet.gender,
            age=pet.age,
            breed=max_confidence_breed or pet.breed,
            color=pet.color
        )
        db.add(new_pet)
        db.commit()
        db.refresh(new_pet)

        new_announcement = Announcement(
            user_id=current_user_id,
            pet_id=new_pet.id,
            keywords=announcement.keywords,
            description=announcement.description,
            location=announcement.location,
            status="published",
            timestamp=datetime.utcnow().isoformat(),
            image_paths=json.dumps(image_urls)
        )
        db.add(new_announcement)
        db.commit()
        db.refresh(new_announcement)

        response = AnnouncementResponse(
            id=new_announcement.id,
            user_id=new_announcement.user_id,
            pet_id=new_announcement.pet_id,
            keywords=new_announcement.keywords,
            description=new_announcement.description,
            location=new_announcement.location,
            status=new_announcement.status,
            timestamp=new_announcement.timestamp,
            image_paths=json.loads(new_announcement.image_paths),
            user=user,
            pet=new_pet
        )
        logger.info(f"Announcement created: id={new_announcement.id}, pet={new_pet.name or 'No name'}")
        return response
    except HTTPException as e:
        for file_url in image_urls:
            await delete_from_gcs(file_url)
        raise e
    except Exception as e:
        for file_url in image_urls:
            await delete_from_gcs(file_url)
        logger.error(f"Error creating announcement: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/update_announcement/{announcement_id}", response_model=AnnouncementResponse)
async def update_announcement(
    announcement_id: int,
    pet: PetCreate = Depends(pet_form),
    announcement: AnnouncementCreate = Depends(announcement_form),
    images: List[UploadFile] = File([]),
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Update an existing announcement."""
    logger.info(f"Updating announcement id={announcement_id} for user_id: {current_user_id}")
    existing_announcement = db.query(Announcement).filter(
        Announcement.id == announcement_id,
        Announcement.user_id == current_user_id
    ).first()
    if not existing_announcement:
        raise HTTPException(status_code=404, detail="Announcement not found or you are not the owner")

    existing_pet = db.query(Pet).filter(Pet.id == existing_announcement.pet_id).first()
    if not existing_pet:
        raise HTTPException(status_code=404, detail="Associated pet not found")

    new_image_urls = []
    old_image_urls = json.loads(existing_announcement.image_paths) if existing_announcement.image_paths else []
    try:
        image_data_pairs = []
        for image in images:
            ext = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
            image_data = await image.read()
            if not image_data:
                raise HTTPException(status_code=400, detail=f"Empty image: {image.filename}")
            image_data_pairs.append((image, image_data))

        suggested_animal_type = None
        max_confidence_breed = None
        if image_data_pairs:
            suggested_animal_type, _, max_confidence_breed = await process_images_for_animal_type(
                image_data_pairs, for_creation=True
            )
            if suggested_animal_type != pet.animal_type.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Specified animal type '{pet.animal_type}' does not match detected '{suggested_animal_type}'"
                )
            for image, image_data in image_data_pairs:
                ext = image.filename.split('.')[-1].lower() if '.' in image.filename else 'jpg'
                file_name = f"{uuid.uuid4()}.{ext}"
                file_url = await upload_to_gcs(image_data, file_name)
                new_image_urls.append(file_url)
        else:
            new_image_urls = old_image_urls

        if not new_image_urls:
            raise HTTPException(status_code=400, detail="At least one image is required")

        full_text = " ".join(filter(None, [
            pet.name, pet.breed, pet.color, str(pet.age) if pet.age is not None else "",
            announcement.description, announcement.keywords, announcement.location or ""
        ]))
        if full_text and check_text_for_toxicity(full_text):
            raise HTTPException(status_code=400, detail="Announcement rejected due to policy violation")

        existing_pet.animal_type = pet.animal_type
        existing_pet.name = pet.name
        existing_pet.gender = pet.gender
        existing_pet.age = pet.age
        existing_pet.breed = max_confidence_breed or pet.breed
        existing_pet.color = pet.color
        db.commit()
        db.refresh(existing_pet)

        existing_announcement.keywords = announcement.keywords
        existing_announcement.description = announcement.description
        existing_announcement.location = announcement.location
        existing_announcement.timestamp = datetime.utcnow().isoformat()
        existing_announcement.image_paths = json.dumps(new_image_urls)
        db.commit()
        db.refresh(existing_announcement)

        if new_image_urls != old_image_urls:
            for old_url in old_image_urls:
                if old_url not in new_image_urls:
                    await delete_from_gcs(old_url)

        response = AnnouncementResponse(
            id=existing_announcement.id,
            user_id=existing_announcement.user_id,
            pet_id=existing_announcement.pet_id,
            keywords=existing_announcement.keywords,
            description=existing_announcement.description,
            location=existing_announcement.location,
            status=existing_announcement.status,
            timestamp=existing_announcement.timestamp,
            image_paths=json.loads(existing_announcement.image_paths),
            user=existing_announcement.user,
            pet=existing_pet
        )
        logger.info(f"Announcement updated: id={existing_announcement.id}, pet={existing_pet.name or 'No name'}")
        return response
    except HTTPException as e:
        for file_url in new_image_urls:
            await delete_from_gcs(file_url)
        raise e
    except Exception as e:
        for file_url in new_image_urls:
            await delete_from_gcs(file_url)
        logger.error(f"Error updating announcement: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/search_announcements", response_model=List[AnnouncementResponse])
async def search_announcements(
    animal_type: Optional[str] = Form(None, description="Animal type (cat or dog)"),
    gender: Optional[str] = Form(None, description="Gender (M or F)"),
    age: Optional[int] = Form(None, description="Animal age"),
    breeds: Optional[str] = Form(None, description="Breeds (comma-separated)"),
    color: Optional[str] = Form(None, description="Animal color"),
    keywords: Optional[str] = Form(None, description="Keywords (comma-separated)"),
    location: Optional[str] = Form(None, description="Animal location"),
    images: List[UploadFile] = File([]),
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Search announcements by criteria."""
    logger.info(f"Searching announcements for user_id: {current_user_id}, parameters: {locals()}")
    try:
        user = db.query(UserModel).filter(UserModel.id == current_user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        breeds_list = [b.strip().lower() for b in breeds.split(",")] if breeds and isinstance(breeds, str) else None
        if breeds_list and len(breeds_list) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 breeds allowed for search")
        keywords_list = [k.strip().lower() for k in keywords.split(",")] if keywords and isinstance(keywords, str) else None

        search = SearchRequest(
            animal_type=animal_type,
            gender=gender,
            age=age,
            breeds=breeds_list,
            color=color,
            keywords=keywords_list,
            location=location
        )

        suggested_animal_type = None
        suggested_breeds = []
        use_vision_api_mapping = False

        if images:
            image_data_pairs = []
            for image in images:
                ext = image.filename.split('.')[-1].lower() if '.' in image.filename else ''
                if ext not in ALLOWED_IMAGE_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
                image_data = await image.read()
                if not image_data:
                    raise HTTPException(status_code=400, detail=f"Empty image: {image.filename}")
                image_data_pairs.append((image, image_data))

            suggested_animal_type, suggested_breeds, _ = await process_images_for_animal_type(
                image_data_pairs, for_creation=False
            )
            use_vision_api_mapping = True
            if suggested_breeds and not search.breeds:
                search.breeds = suggested_breeds[:3]
                logger.info(f"Using breeds from images: {search.breeds}")

        animal_type = search.animal_type or suggested_animal_type
        query = db.query(Announcement).join(Pet).filter(Announcement.status == "published")
        if animal_type:
            query = query.filter(Pet.animal_type.ilike(f"%{animal_type.lower()}%"))

        announcements = query.all()
        if not announcements:
            logger.info("No announcements found after initial filtering")
            return []

        ranked_announcements = []
        for ann in announcements:
            score = 0
            pet = ann.pet

            if search.breeds and pet.breed:
                for breed in search.breeds:
                    if use_vision_api_mapping:
                        if breed.lower() == pet.breed.lower():
                            score += 10
                    else:
                        if breed.lower() in pet.breed.lower() or pet.breed.lower() in breed.lower():
                            score += 10

            if search.keywords and ann.keywords:
                request_keywords = set(kw.lower() for kw in search.keywords)
                ann_keywords = set(kw.lower() for kw in (ann.keywords or "").split(",") if ann.keywords)
                score += len(request_keywords.intersection(ann_keywords)) * 6

            if search.location and ann.location and search.location.lower() in ann.location.lower():
                score += 10

            if search.gender and pet.gender and search.gender.lower() == pet.gender.lower():
                score += 5
            if search.color and pet.color and search.color.lower() == pet.color.lower():
                score += 4
            if search.age is not None and pet.age == search.age:
                score += 3

            ranked_announcements.append((score, ann))

        result = [
            AnnouncementResponse(
                id=ann.id,
                user_id=ann.user_id,
                pet_id=ann.pet_id,
                keywords=ann.keywords,
                description=ann.description,
                location=ann.location,
                status=ann.status,
                timestamp=ann.timestamp,
                image_paths=json.loads(ann.image_paths) if ann.image_paths else [],
                user=ann.user,
                pet=ann.pet
            )
            for _, ann in sorted(ranked_announcements, key=lambda x: (x[0], x[1].timestamp), reverse=True)
        ]
        logger.info(f"Returned {len(result)} announcements")
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error searching announcements: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/announcements/my", response_model=List[AnnouncementResponse])
async def get_my_announcements(
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Get list of current user's announcements."""
    logger.info(f"Getting my announcements for user_id: {current_user_id}")
    try:
        announcements = db.query(Announcement).filter(
            Announcement.user_id == current_user_id,
            Announcement.status == "published"
        ).all()

        if not announcements:
            logger.info(f"No announcements found for user_id: {current_user_id}")
            return []

        response = [
            AnnouncementResponse(
                id=ann.id,
                user_id=ann.user_id,
                pet_id=ann.pet_id,
                keywords=ann.keywords,
                description=ann.description,
                location=ann.location,
                status=ann.status,
                timestamp=ann.timestamp,
                image_paths=json.loads(ann.image_paths) if ann.image_paths else [],
                user=ann.user,
                pet=ann.pet
            )
            for ann in announcements
        ]
        logger.info(f"Returned {len(response)} announcements for user_id: {current_user_id}")
        return response
    except Exception as e:
        logger.error(f"Error getting my announcements: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.delete("/announcements/{announcement_id}")
async def delete_announcement(
    announcement_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Delete an announcement."""
    logger.info(f"Deleting announcement id={announcement_id} for user_id: {current_user_id}")
    try:
        announcement = db.query(Announcement).filter(
            Announcement.id == announcement_id,
            Announcement.user_id == current_user_id
        ).first()
        if not announcement:
            raise HTTPException(status_code=404, detail="Announcement not found or you are not the owner")

        pet = db.query(Pet).filter(Pet.id == announcement.pet_id).first()
        if not pet:
            raise HTTPException(status_code=404, detail="Associated pet not found")

        image_urls = json.loads(announcement.image_paths) if announcement.image_paths else []
        for image_url in image_urls:
            await delete_from_gcs(image_url)

        db.delete(announcement)
        db.delete(pet)
        db.commit()

        logger.info(f"Announcement deleted: id={announcement_id}, pet_id={pet.id}")
        return {"message": "Announcement deleted"}
    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting announcement: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/favorites", response_model=FavoriteResponse)
async def add_favorite(
    favorite: FavoriteCreate,
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Add announcement to favorites."""
    logger.info(f"Adding to favorites for user_id: {current_user_id}, announcement_id: {favorite.announcement_id}")
    announcement = db.query(Announcement).filter(Announcement.id == favorite.announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    existing_favorite = db.query(Favorite).filter(
        Favorite.user_id == current_user_id,
        Favorite.announcement_id == favorite.announcement_id
    ).first()
    if existing_favorite:
        raise HTTPException(status_code=400, detail="Announcement already in favorites")

    favorite_count = db.query(Favorite).filter(Favorite.user_id == current_user_id).count()
    if favorite_count >= 10:
        raise HTTPException(status_code=400, detail="Maximum 10 favorite announcements allowed")

    db_favorite = Favorite(user_id=current_user_id, announcement_id=favorite.announcement_id)
    db.add(db_favorite)
    try:
        db.commit()
        db.refresh(db_favorite)
        logger.info(f"Favorite added: user_id={current_user_id}, announcement_id={favorite.announcement_id}")
        return FavoriteResponse(user_id=db_favorite.user_id, announcement_id=db_favorite.announcement_id)
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding to favorites: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@router.delete("/favorites/{announcement_id}")
async def remove_favorite(
    announcement_id: int,
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Remove announcement from favorites."""
    logger.info(f"Removing from favorites for user_id: {current_user_id}, announcement_id: {announcement_id}")
    db_favorite = db.query(Favorite).filter(
        Favorite.user_id == current_user_id,
        Favorite.announcement_id == announcement_id
    ).first()
    if not db_favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")

    db.delete(db_favorite)
    try:
        db.commit()
        logger.info(f"Favorite removed: user_id={current_user_id}, announcement_id={announcement_id}")
        return {"message": "Favorite removed"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing from favorites: {e}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@router.get("/favorites", response_model=List[AnnouncementResponse])
async def get_favorites(
    db: Session = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    """Get list of favorite announcements."""
    logger.info(f"Getting favorites for user_id: {current_user_id}")
    favorites = db.query(Favorite).filter(Favorite.user_id == current_user_id).all()
    if not favorites:
        logger.info(f"No favorites found for user_id: {current_user_id}")
        return []

    announcement_ids = [f.announcement_id for f in favorites]
    announcements = db.query(Announcement).filter(
        Announcement.id.in_(announcement_ids),
        Announcement.status == "published"
    ).all()

    response = [
        AnnouncementResponse(
            id=ann.id,
            user_id=ann.user_id,
            pet_id=ann.pet_id,
            keywords=ann.keywords,
            description=ann.description,
            location=ann.location,
            status=ann.status,
            timestamp=ann.timestamp,
            image_paths=json.loads(ann.image_paths) if ann.image_paths else [],
            user=ann.user,
            pet=ann.pet
        )
        for ann in announcements
    ]
    logger.info(f"Returned {len(response)} favorite announcements for user_id: {current_user_id}")
    return response
