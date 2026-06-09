from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TOKEN")
if not TOKEN:
    logger.warning("TOKEN environment variable not found! Set TOKEN before starting.")
    TOKEN = "YOUR_HARDCODED_TOKEN_HERE"  # fallback — replace with actual token if env var is unavailable

logger.info("TOKEN loaded. Starting bot...")

# Lazy-loaded pipeline — only initialised on the first photo request
pipe = None

def load_pipeline():
    """Load the Stable Diffusion img2img pipeline on first use (lazy loading)."""
    global pipe
    if pipe is None:
        logger.info("Loading StableDiffusionImg2ImgPipeline for the first time...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "Lykon/dreamshaper-8",
            torch_dtype=torch.float32,
        )
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()  # reduces memory usage by ~50%
        logger.info("Pipeline loaded successfully.")
    return pipe


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Photo received. Generating nude version... (may take 30-60 seconds)"
    )

    photo = await update.message.photo[-1].get_file()
    photo_bytes = await photo.download_as_bytearray()

    init_image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = (
        "completely nude, naked, no clothes, detailed skin texture, "
        "realistic body, natural lighting"
    )
    negative = (
        "clothes, dress, shirt, pants, underwear, bikini, bra, censored, "
        "blurry, deformed, low quality, text, watermark"
    )

    pipeline = load_pipeline()
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        image=init_image,
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=35,
    ).images[0]

    # Free GPU/CPU memory after generation
    torch.cuda.empty_cache()

    img_byte = io.BytesIO()
    image.save(img_byte, format="PNG")
    img_byte.seek(0)

    await update.message.reply_photo(photo=img_byte, caption="Yeh lo bhai 🔥")


app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

logger.info("🤖 Bot is running...")

if __name__ == "__main__":
    app.run_polling()
