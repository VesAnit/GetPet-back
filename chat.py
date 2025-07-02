from fastapi import APIRouter, WebSocket, Depends, HTTPException, Form
from sqlalchemy.orm import Session, joinedload
from datetime import datetime
from typing import List, Dict
from models import User, MessageModel, Announcement, Chat, ChatParticipant
from database import get_db
from auth import get_current_user
import json
import logging

router = APIRouter(prefix="", tags=["chat"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connected_clients: Dict[int, WebSocket] = {}

@router.websocket("/ws/chat/{announcement_id}")
async def websocket_endpoint(websocket: WebSocket, announcement_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        logger.warning(f"Announcement not found: announcement_id={announcement_id}")
        await websocket.close(code=1008, reason="Announcement not found")
        return

    sender_id = user.id
    receiver_id = announcement.user_id

    chat = db.query(Chat).filter(Chat.announcement_id == announcement_id).first()
    if not chat:
        if sender_id == receiver_id:
            logger.warning(f"User {sender_id} cannot initiate chat with their own announcement")
            await websocket.close(code=1008, reason="Cannot initiate chat with your own announcement")
            return
        # новый чат с участниками
        chat = Chat(announcement_id=announcement_id)
        db.add(chat)
        db.flush()
        db.add(ChatParticipant(chat_id=chat.id, user_id=receiver_id))
        db.add(ChatParticipant(chat_id=chat.id, user_id=sender_id))
        db.commit()
        logger.info(f"Created new chat {chat.id} for announcement {announcement_id}")
    else:
        # если чат существует, проверяем, участвует ли текущий пользователь
        is_participant = db.query(ChatParticipant).filter(
            ChatParticipant.chat_id == chat.id,
            ChatParticipant.user_id == sender_id
        ).first() is not None
        if not is_participant:
            logger.warning(f"User {sender_id} is not a participant of chat {chat.id}")
            await websocket.close(code=1008, reason="You are not a participant of this chat")
            return

    await websocket.accept()
    connected_clients[sender_id] = websocket
    logger.info(f"User {sender_id} connected to WebSocket for chat {chat.id}")

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            new_message = MessageModel(
                chat_id=chat.id,
                sender_id=sender_id,
                content=message_data["message"],
                timestamp=datetime.now().isoformat(),
                is_read=False  # Новое сообщение по умолчанию непрочитанное
            )
            db.add(new_message)
            db.commit()
            db.refresh(new_message)
            logger.info(f"Message saved (WebSocket): from {sender_id} to chat {chat.id}, content: {message_data['message']}")

            message_response = {
                "id": new_message.id,
                "chat_id": chat.id,
                "sender_id": sender_id,
                "content": message_data["message"],
                "timestamp": new_message.timestamp,
                "is_read": new_message.is_read,
                "sender": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email
                }
            }

            for participant in chat.participants:
                participant_id = participant.id
                if participant_id in connected_clients:
                    await connected_clients[participant_id].send_text(json.dumps(message_response))
                    logger.info(f"Message sent to participant {participant_id} via WebSocket")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if sender_id in connected_clients:
            del connected_clients[sender_id]
            logger.info(f"User {sender_id} disconnected from WebSocket")
        await websocket.close()

@router.post("/chat/{announcement_id}")
async def send_message(
    announcement_id: int,
    message: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    sender_id = user.id
    receiver_id = announcement.user_id

    chat = db.query(Chat).filter(Chat.announcement_id == announcement_id).first()
    if not chat:

        if sender_id == receiver_id:
            raise HTTPException(status_code=400, detail="Cannot initiate chat with your own announcement")
        # Создаём новый чат с участниками
        chat = Chat(announcement_id=announcement_id)
        db.add(chat)
        db.flush()
        db.add(ChatParticipant(chat_id=chat.id, user_id=receiver_id))
        db.add(ChatParticipant(chat_id=chat.id, user_id=sender_id))
        db.commit()
        logger.info(f"Created new chat {chat.id} for announcement {announcement_id}")
    else:
        is_participant = db.query(ChatParticipant).filter(
            ChatParticipant.chat_id == chat.id,
            ChatParticipant.user_id == sender_id
        ).first() is not None
        if not is_participant:
            raise HTTPException(status_code=403, detail="You are not a participant of this chat")

    new_message = MessageModel(
        chat_id=chat.id,
        sender_id=sender_id,
        content=message,
        timestamp=datetime.now().isoformat(),
        is_read=False  # Новое сообщение по умолчанию непрочитанное
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    logger.info(f"Message saved (REST): from {sender_id} to chat {chat.id}, content: {message}")

    message_response = {
        "id": new_message.id,
        "chat_id": chat.id,
        "sender_id": sender_id,
        "content": message,
        "timestamp": new_message.timestamp,
        "is_read": new_message.is_read,
        "sender": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    }

    for participant in chat.participants:
        participant_id = participant.id
        if participant_id in connected_clients:
            await connected_clients[participant_id].send_text(json.dumps(message_response))
            logger.info(f"Message sent to participant {participant_id} via WebSocket (REST)")

    return {"status": "Message sent"}

@router.get("/chat/{announcement_id}")
def get_chat(
    announcement_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found")

    user_id = user.id

    chat = db.query(Chat).filter(Chat.announcement_id == announcement_id).first()
    if not chat:
        logger.info(f"No chat found for announcement {announcement_id}, user {user_id} can initiate")
        return []

    is_participant = db.query(ChatParticipant).filter(
        ChatParticipant.chat_id == chat.id,
        ChatParticipant.user_id == user_id
    ).first() is not None
    is_owner = (announcement.user_id == user_id)

    if not (is_participant or is_owner):
        raise HTTPException(status_code=403, detail="You are not authorized to view this chat")

    messages = db.query(MessageModel).options(
        joinedload(MessageModel.sender)
    ).filter(MessageModel.chat_id == chat.id).order_by(MessageModel.timestamp.asc()).all()

    if not messages:
        logger.info(f"No messages found for chat {chat.id}, user {user_id}")
        return []

    result = [
        {
            "id": msg.id,
            "chat_id": msg.chat_id,
            "sender_id": msg.sender_id,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "is_read": msg.is_read,
            "sender": {
                "id": msg.sender.id,
                "username": msg.sender.username,
                "email": msg.sender.email
            } if msg.sender else None
        }
        for msg in messages
    ]
    logger.info(f"Retrieved {len(messages)} messages for user {user_id} from chat {chat.id}")
    return result

@router.get("/chats")
def get_user_chats(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_id = user.id

    chats = db.query(Chat).join(ChatParticipant).filter(
        ChatParticipant.user_id == user_id
    ).options(
        joinedload(Chat.announcement).joinedload(Announcement.pet),
        joinedload(Chat.participants),
        joinedload(Chat.messages).joinedload(MessageModel.sender)
    ).all()

    if not chats:
        logger.info(f"No chats found for user {user_id}")
        return []

    result = []
    for chat in chats:
        # Сортируем сообщения по timestamp и берём последнее
        messages = sorted(chat.messages, key=lambda m: m.timestamp) if chat.messages else []
        last_message = messages[-1] if messages else None
        if last_message:
            logger.debug(f"Chat {chat.id}: Last message timestamp: {last_message.timestamp}")

        other_participant = next(
            (participant for participant in chat.participants if participant.id != user_id),
            None
        )

        image_paths = []
        try:
            if chat.announcement and chat.announcement.image_paths:
                image_paths = json.loads(chat.announcement.image_paths)
                if not isinstance(image_paths, list):
                    logger.warning(f"Invalid image_paths format for chat {chat.id}: {chat.announcement.image_paths}")
                    image_paths = []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for image_paths in chat {chat.id}: {e}")
            image_paths = []

        unread_count = len([
            msg for msg in messages
            if not msg.is_read and msg.sender_id != user_id
        ])
        has_unread = unread_count > 0

        chat_info = {
            "chat_id": chat.id,
            "announcement_id": chat.announcement_id,
            "announcement_title": chat.announcement.pet.name if chat.announcement and chat.announcement.pet else f"Announcement {chat.announcement_id}",
            "image_paths": image_paths,
            "has_unread": has_unread,  # Флаг наличия непрочитанных сообщений
            "other_participant": {
                "id": other_participant.id,
                "username": other_participant.username,
                "email": other_participant.email
            } if other_participant else None,
            "last_message": {
                "id": last_message.id,
                "sender_id": last_message.sender_id,
                "content": last_message.content,
                "timestamp": last_message.timestamp,
                "is_read": last_message.is_read,
                "sender": {
                    "id": last_message.sender.id,
                    "username": last_message.sender.username,
                    "email": last_message.sender.email
                } if last_message and last_message.sender else None
            } if last_message else None
        }
        result.append(chat_info)

    logger.info(f"Retrieved {len(result)} chats for user {user_id}")
    return result

@router.post("/chat/{announcement_id}/mark_read")
async def mark_messages_as_read(
    announcement_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.announcement_id == announcement_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_id = user.id

    is_participant = db.query(ChatParticipant).filter(
        ChatParticipant.chat_id == chat.id,
        ChatParticipant.user_id == user_id
    ).first() is not None
    if not is_participant:
        raise HTTPException(status_code=403, detail="You are not a participant of this chat")

    unread_messages = db.query(MessageModel).filter(
        MessageModel.chat_id == chat.id,
        MessageModel.is_read == False,
        MessageModel.sender_id != user_id
    ).all()

    if unread_messages:
        db.query(MessageModel).filter(
            MessageModel.chat_id == chat.id,
            MessageModel.is_read == False,
            MessageModel.sender_id != user_id
        ).update({MessageModel.is_read: True})
        db.commit()
        logger.info(f"Updated messages to is_read=True for chat {chat.id}")
    else:
        logger.info(f"No messages to mark as read for chat {chat.id}")

    db.expire_all()
    return {"status": "Messages marked as read"}
