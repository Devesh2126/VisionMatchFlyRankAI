"""
Root runner for VisionMatch AI Benchmark Seeder.
"""
import io
import sys
from pathlib import Path

# Force Python absolute path resolution for backend
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = (ROOT / "backend").resolve()
ML_DIR = (ROOT / "ml").resolve()

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))

from PIL import Image, ImageDraw
from sqlalchemy import Column, Text, select, text

# CORRECTED IMPORTS TO MATCH YOUR ZIP FILE STRUCTURE
from app.core.security import hash_password
from app.db.session import SessionLocal, init_db
from app.db.models import ImageRecord, MLModelRecord, User
from app.services.storage_service import storage_service
from app.workers.tasks import (
    execute_evaluation_benchmark,
    process_image_embedding_task,
)
from visionmatch_ml.preprocessing import create_thumbnail_bytes, load_and_sanitize_image

# The benchmark needs a caption per image; add the column to the mapping if the model lacks it.
if not hasattr(ImageRecord, "caption_text"):
    ImageRecord.caption_text = Column(Text, nullable=True)

BENCHMARK_ITEMS = [
    {
        "filename": "red_sports_car_sunset_beach.png",
        "category": "automotive",
        "caption": "A bright red sports car parked beside a sunset ocean beach with orange sky",
        "bg": (25, 35, 75),
        "shape": "car_sunset",
    },
    {
        "filename": "crimson_coupe_coastal_highway.png",
        "category": "automotive",
        "caption": "A red racing car on a coastal road at dusk",
        "bg": (40, 25, 65),
        "shape": "car_sunset",
    },
    {
        "filename": "alpine_snowy_mountain_peaks.png",
        "category": "landscape",
        "caption": "Snow-capped mountain peaks under a clear blue alpine sky",
        "bg": (65, 135, 225),
        "shape": "mountains",
    },
    {
        "filename": "highland_mountain_ridge_valley.png",
        "category": "landscape",
        "caption": "Majestic rocky mountain range with white snow summits and green valley",
        "bg": (55, 120, 210),
        "shape": "mountains",
    },
    {
        "filename": "golden_retriever_sunny_meadow.png",
        "category": "animals",
        "caption": "A golden retriever dog sitting on green grass in a sunny park",
        "bg": (60, 165, 85),
        "shape": "golden_dog",
    },
    {
        "filename": "yellow_labrador_park_lawn.png",
        "category": "animals",
        "caption": "A happy golden yellow dog outdoors on a bright green lawn",
        "bg": (50, 155, 75),
        "shape": "golden_dog",
    },
    {
        "filename": "neon_cyberpunk_night_skyline.png",
        "category": "urban",
        "caption": "Futuristic cyberpunk city street at night with glowing neon pink and cyan skyscrapers",
        "bg": (12, 12, 32),
        "shape": "neon_city",
    },
    {
        "filename": "downtown_metropolis_night_lights.png",
        "category": "urban",
        "caption": "Dark metropolitan skyline illuminated by bright neon towers at midnight",
        "bg": (18, 15, 40),
        "shape": "neon_city",
    },
    {
        "filename": "tropical_emerald_forest_canopy.png",
        "category": "nature",
        "caption": "Dense green forest trees and lush botanical foliage in sunlight",
        "bg": (25, 95, 45),
        "shape": "forest",
    },
    {
        "filename": "woodland_pine_trees_trail.png",
        "category": "nature",
        "caption": "Tall green pine trees in a quiet emerald woodland forest",
        "bg": (30, 105, 55),
        "shape": "forest",
    },
]


def render_scene_bytes(bg: tuple[int, int, int], shape: str, seed_offset: int) -> bytes:
    img = Image.new("RGB", (384, 384), color=bg)
    draw = ImageDraw.Draw(img)

    if shape == "car_sunset":
        draw.rectangle([0, 0, 384, 210], fill=(245, 115, 55))
        draw.ellipse([140 + seed_offset, 90, 240 + seed_offset, 190], fill=(255, 210, 70))
        draw.rectangle([0, 210, 384, 280], fill=(30, 85, 160))
        draw.rectangle([0, 280, 384, 384], fill=(215, 190, 140))
        draw.polygon([(70, 310), (120, 270), (250, 270), (310, 310)], fill=(225, 25, 35))
        draw.rectangle([55, 305, 325, 345], fill=(235, 30, 40))
        draw.ellipse([90, 325, 140, 375], fill=(25, 25, 25))
        draw.ellipse([240, 325, 290, 375], fill=(25, 25, 25))
    elif shape == "mountains":
        draw.polygon([(20, 320), (140 + seed_offset, 90), (260, 320)], fill=(95, 105, 125))
        draw.polygon([(110 + seed_offset, 145), (140 + seed_offset, 90), (170 + seed_offset, 145)], fill=(250, 252, 255))
        draw.polygon([(150, 320), (275, 110), (375, 320)], fill=(75, 85, 105))
        draw.polygon([(245, 165), (275, 110), (305, 165)], fill=(250, 252, 255))
        draw.rectangle([0, 310, 384, 384], fill=(45, 140, 70))
    elif shape == "golden_dog":
        draw.rectangle([0, 0, 384, 180], fill=(135, 205, 250))
        draw.rectangle([0, 180, 384, 384], fill=(65, 175, 75))
        draw.ellipse([125 + seed_offset, 195, 265 + seed_offset, 320], fill=(230, 180, 75))
        draw.ellipse([155 + seed_offset, 125, 235 + seed_offset, 210], fill=(240, 190, 85))
        draw.ellipse([185 + seed_offset, 155, 205 + seed_offset, 175], fill=(30, 25, 20))
    elif shape == "neon_city":
        for idx, x in enumerate(range(25, 340, 60)):
            h = 110 + ((idx * 37 + seed_offset) % 140)
            draw.rectangle([x, h, x + 46, 360], fill=(28, 32, 64))
            neon = (255, 40, 180) if idx % 2 == 0 else (30, 235, 255)
            draw.rectangle([x + 6, h + 10, x + 40, h + 22], fill=neon)
            draw.rectangle([x + 18, h + 35, x + 28, 350], fill=neon)
    elif shape == "forest":
        draw.rectangle([0, 300, 384, 384], fill=(65, 50, 35))
        for x in range(45, 350, 75):
            draw.rectangle([x + 22, 240, x + 38, 330], fill=(95, 65, 35))
            draw.polygon([(x - 15, 250), (x + 30, 95 + seed_offset), (x + 75, 250)], fill=(35, 145, 65))
            draw.polygon([(x - 5, 185), (x + 30, 65 + seed_offset), (x + 65, 185)], fill=(50, 175, 80))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_schema() -> None:
    """Create the pgvector extension and all tables BEFORE init_db() seeds data."""
    with SessionLocal() as db:
        engine = db.get_bind()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # All declarative models share one metadata object, so this creates every table.
    MLModelRecord.metadata.create_all(bind=engine)
    # init_db() inserts the CLIP row without architecture_family; allow NULL so it can't fail.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE ml_models ALTER COLUMN architecture_family DROP NOT NULL"))
        conn.execute(
            text(f'ALTER TABLE "{ImageRecord.__tablename__}" ADD COLUMN IF NOT EXISTS caption_text TEXT')
        )


def main() -> None:
    print("Creating pgvector extension & tables...")
    create_schema()

    print("Initializing database & model registry...")
    init_db()

    with SessionLocal() as db:
        demo_email = "demo@visionmatch.ai"
        user = db.scalar(select(User).where(User.email == demo_email))
        if not user:
            user = User(
                email=demo_email,
                hashed_password=hash_password("VisionMatch123!"),
                display_name="Demo Admin",
                is_superuser=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created demo account: {demo_email} / VisionMatch123!")

        default_model = db.scalar(
            select(MLModelRecord).where(MLModelRecord.model_key == "clip-vit-b32")
        )
        assert default_model is not None, "Default model clip-vit-b32 not found in database registry."

        for idx, item in enumerate(BENCHMARK_ITEMS):
            raw_bytes = render_scene_bytes(item["bg"], item["shape"], seed_offset=(idx % 3) * 8)
            sanitized = load_and_sanitize_image(raw_bytes)

            existing = db.scalar(
                select(ImageRecord).where(
                    ImageRecord.sha256_checksum == sanitized.sha256_checksum
                )
            )
            if existing:
                print(f"Skipping duplicate benchmark item: {item['filename']}")
                continue

            img_rec = ImageRecord(
                owner_id=user.id,
                storage_path=f"benchmark/originals/{item['filename']}",
                thumbnail_path=f"benchmark/thumbnails/{item['filename']}.webp",
                filename=item["filename"],
                mime_type=sanitized.mime_type,
                file_size_bytes=sanitized.file_size_bytes,
                width=sanitized.width,
                height=sanitized.height,
                sha256_checksum=sanitized.sha256_checksum,
            )
            storage_service.upload_bytes(
                img_rec.storage_path, sanitized.clean_bytes, sanitized.mime_type
            )
            storage_service.upload_bytes(
                img_rec.thumbnail_path,
                create_thumbnail_bytes(sanitized.pil_image),
                "image/webp",
            )
            db.add(img_rec)
            db.commit()
            db.refresh(img_rec)

            print(f"Encoding benchmark image [{idx+1}/{len(BENCHMARK_ITEMS)}]: {item['filename']}")
            process_image_embedding_task(str(img_rec.id), "clip-vit-b32", None)

        for item in BENCHMARK_ITEMS:
            rec = db.scalar(select(ImageRecord).where(ImageRecord.filename == item["filename"]))
            if rec is not None:
                rec.caption_text = item["caption"]
        db.commit()

        print("Running baseline IR evaluation for 'clip-vit-b32'...")
        run1 = execute_evaluation_benchmark(
            db, "clip-vit-b32", "curated_multimodal_benchmark_v1"
        )
        print(
            f"Evaluation complete -> Recall@1: {run1.recall_at_1}, Recall@5: {run1.recall_at_5}, "
            f"mAP: {run1.mean_average_precision}, MRR: {run1.mean_reciprocal_rank}"
        )


if __name__ == "__main__":
    main()