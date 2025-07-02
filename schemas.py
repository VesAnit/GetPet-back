from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class PetCreate(BaseModel):
    animal_type: str
    name: Optional[str] = None
    gender: str
    age: Optional[int] = None
    breed: Optional[str] = None
    color: Optional[str] = None

class Pet(BaseModel):
    id: int
    animal_type: str
    name: Optional[str] = None
    gender: str
    age: Optional[int] = None
    breed: Optional[str] = None
    color: Optional[str] = None

    class Config:
        from_attributes = True

class AnnouncementCreate(BaseModel):
    keywords: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None  # Добавлено поле

class AnnouncementResponse(BaseModel):
    id: int
    user_id: int
    pet_id: int
    keywords: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None  # Добавлено поле
    status: str
    timestamp: str
    image_paths: List[str]
    user: User
    pet: Pet

    class Config:
        from_attributes = True

class SearchRequest(BaseModel):
    animal_type: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    breeds: Optional[List[str]] = None
    color: Optional[str] = None
    keywords: Optional[List[str]] = None
    location: Optional[str] = None  # Добавлено поле

class MessageCreate(BaseModel):
    sender_id: int
    content: str  # Убраны receiver_id и announcement_id, так как они определяются через chat
    timestamp: str

class Message(BaseModel):
    id: int
    chat_id: int  # Новое поле
    sender_id: int
    content: str
    timestamp: str
    is_read: bool  # Новое поле

    class Config:
        from_attributes = True

class ValidateImageResponse(BaseModel):
    suggested_animal_type: Optional[str]
    max_confidence_breed: Optional[str]
    confidence: float
    error_message: Optional[str]
    invalid_image_index: Optional[int]

class FavoriteCreate(BaseModel):
    announcement_id: int

class FavoriteResponse(BaseModel):
    user_id: int
    announcement_id: int

    class Config:
        from_attributes = True
