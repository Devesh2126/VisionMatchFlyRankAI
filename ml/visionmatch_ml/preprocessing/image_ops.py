import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

# Security: Protect against decompression bomb DOS attacks (max 25 megapixels)
Image.MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidationError(ValueError):
    """Raised when an uploaded file fails security or format validation."""


@dataclass(frozen=True)
class SanitizedImageResult:
    pil_image: Image.Image
    clean_bytes: bytes
    mime_type: str
    width: int
    height: int
    file_size_bytes: int
    sha256_checksum: str


def load_and_sanitize_image(
    raw_bytes: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> SanitizedImageResult:
    """
    Validates raw image bytes, applies EXIF orientation,
    strips metadata and converts the image to RGB.
    """
    if not raw_bytes:
        raise ImageValidationError("Empty file payload received.")

    if len(raw_bytes) > max_size_bytes:
        raise ImageValidationError(
            f"File size ({len(raw_bytes)} bytes) exceeds limit ({max_size_bytes} bytes)."
        )

    sha256_checksum = hashlib.sha256(raw_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_format = (img.format or "").upper()

            if img_format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported image format '{img_format}'. "
                    f"Allowed: {sorted(ALLOWED_FORMATS)}"
                )

            img.load()

            transposed = ImageOps.exif_transpose(img)

            if transposed.mode in ("RGBA", "LA") or (
                transposed.mode == "P" and "transparency" in transposed.info
            ):
                background = Image.new("RGB", transposed.size, (255, 255, 255))
                alpha = transposed.convert("RGBA").split()[-1]
                background.paste(transposed.convert("RGB"), mask=alpha)
                rgb_image = background
            else:
                rgb_image = transposed.convert("RGB")

            width, height = rgb_image.size

            if width < 16 or height < 16:
                raise ImageValidationError(
                    f"Image dimensions ({width}x{height}) are too small (minimum 16x16)."
                )

            output_buffer = io.BytesIO()
            save_format = img_format

            rgb_image.save(output_buffer, format=save_format, quality=92)
            clean_bytes = output_buffer.getvalue()

            return SanitizedImageResult(
                pil_image=rgb_image,
                clean_bytes=clean_bytes,
                mime_type=FORMAT_TO_MIME[save_format],
                width=width,
                height=height,
                file_size_bytes=len(clean_bytes),
                sha256_checksum=sha256_checksum,
            )

    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "Image exceeds maximum safe pixel resolution."
        ) from exc

    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            "File is corrupt or not a valid image."
        ) from exc


def create_thumbnail_bytes(
    pil_image: Image.Image,
    max_dim: int = 400,
) -> bytes:
    """Generates a lightweight WebP thumbnail for fast gallery rendering."""

    thumb = pil_image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    thumb.save(buf, format="WEBP", quality=82)

    return buf.getvalue()