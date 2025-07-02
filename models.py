from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from database import Base
import json
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    announcements = relationship("Announcement", back_populates="user")
    sent_messages = relationship("MessageModel", foreign_keys="MessageModel.sender_id", back_populates="sender")
    favorites = relationship("Favorite", back_populates="user")
    chats = relationship("Chat", secondary="chat_participants", back_populates="participants")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email
        }

class Pet(Base):
    __tablename__ = "pets"
    id = Column(Integer, primary_key=True, index=True)
    animal_type = Column(String)
    name = Column(String, nullable=True)
    gender = Column(String)
    age = Column(Integer, nullable=True)
    breed = Column(String, nullable=True)
    color = Column(String, nullable=True)
    announcements = relationship("Announcement", back_populates="pet")

    def to_dict(self):
        return {
            "id": self.id,
            "animal_type": self.animal_type,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "breed": self.breed,
            "color": self.color
        }

class Announcement(Base):
    __tablename__ = "announcements"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pet_id = Column(Integer, ForeignKey("pets.id"))
    keywords = Column(String, nullable=True)
    description = Column(String, nullable=True)
    status = Column(String)
    timestamp = Column(String)
    location = Column(String, nullable=True)
    image_paths = Column(String)
    user = relationship("User", back_populates="announcements")
    pet = relationship("Pet", back_populates="announcements")
    favorites = relationship("Favorite", back_populates="announcement")
    chat = relationship("Chat", back_populates="announcement", uselist=False)

    def to_dict(self):
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "pet_id": self.pet_id,
            "keywords": self.keywords,
            "description": self.description,
            "status": self.status,
            "timestamp": self.timestamp,
            "location": self.location,
            "image_paths": json.loads(self.image_paths) if self.image_paths else [],
            "user": self.user.to_dict() if self.user else None,
            "pet": self.pet.to_dict() if self.pet else None
        }
        return data

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    announcement_id = Column(Integer, ForeignKey("announcements.id"), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("MessageModel", back_populates="chat")
    participants = relationship("User", secondary="chat_participants", back_populates="chats")
    announcement = relationship("Announcement", back_populates="chat")

class ChatParticipant(Base):
    __tablename__ = "chat_participants"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    user_id = Column(Integer, ForeignKey("users.id"))

class MessageModel(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    sender_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String, nullable=False)
    timestamp = Column(String)
    is_read = Column(Boolean, default=False, nullable=False)  # Новое поле
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_messages")
    chat = relationship("Chat", back_populates="messages")

class Favorite(Base):
    __tablename__ = "favorites"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    announcement_id = Column(Integer, ForeignKey("announcements.id"), primary_key=True)
    user = relationship("User", back_populates="favorites")
    announcement = relationship("Announcement", back_populates="favorites")
