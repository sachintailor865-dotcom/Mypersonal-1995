from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import os
import logging
logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("TOKEN")
if not TOKEN:
 print(":x: TOKEN environment variable not found!")
exit(1)
print(":white_check_mark: TOKEN found. Starting bot...")
# Model (CPU mode for free Railway)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("Lykon/dreamshaper-8",torch_dtype=torch.float32)
pipe = pipe.to("cpu")
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
 await update.message.reply_text("Photo received. Generating nude version... (may take 30-60 seconds)")
 photo = await update.message.photo[-1].get_file()
 photo_bytes = await photo.download_as_bytearray()
 init_image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
 init_image = init_image.resize((512, 512))
 prompt = "completely nude, naked, no clothes, detailed skin texture, realistic body, natural lighting"
 negative = "clothes, dress, shirt, pants, underwear, bikini, bra, censored, blurry, deformed, low quality, text, watermark"
 image = pipe(
 prompt=prompt,
 negative_prompt=negative,
 image=init_image,
 strength=0.75,
 guidance_scale=7.5,
 num_inference_steps=35
 ).images[0]
 img_byte = io.BytesIO()
 image.save(img_byte, format='PNG')
 img_byte.seek(0)
 await update.message.reply_photo(photo=img_byte, caption="Yeh lo bhai :fire:")
app = Application.builder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
print(":robot_face: Bot is running...")
if __name__ == "__main__":
 app.run_polling()
