from fastapi import APIRouter
from app.api.v1.endpoints import auth, images, models, search

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(images.router)
api_router.include_router(search.router)
api_router.include_router(models.router)