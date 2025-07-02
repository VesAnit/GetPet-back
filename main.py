from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
logger.info("Environment variables loaded: %s", os.getenv("DATABASE_URL", "Not set"))

# Проверка наличия модулей
try:
    from database import Base, engine
    from users import router as users_router
    from classify import router as classify_router
    from chat import router as chat_router
except ImportError as e:
    logger.error("Failed to import modules: %s", e)
    raise

app = FastAPI(
    title="Pet Adoption API",
    description="API for pet adoption platform with image classification",
    version="1.0.0",
    openapi_tags=[
        {"name": "users", "description": "Operations with users"},
        {"name": "classify", "description": "Image classification and announcement operations"},
        {"name": "chat", "description": "Chat operations"},
    ],
)

# Создаём таблицы в базе данных
logger.info("Creating database tables...")
Base.metadata.create_all(bind=engine)
logger.info("Database tables created")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#if not os.path.exists("uploads"):
    #logger.info("Creating uploads directory: uploads")
    #os.makedirs("uploads")

#app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
#logger.info("Static files mounted at /uploads")

app.include_router(users_router)
app.include_router(classify_router)
app.include_router(chat_router)
logger.info("Routers included: users, classify, chat")

@app.get("/")
def read_root():
    return {"message": "Pet Adoption API"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unexpected error: %s", str(exc))
    from starlette.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )
