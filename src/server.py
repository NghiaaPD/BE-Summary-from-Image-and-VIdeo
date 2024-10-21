from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
from transformers import AutoProcessor, AutoModelForCausalLM
import os
from PIL import Image

model_id = 'microsoft/Florence-2-base-ft'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
    
images_dir = "./images"

@app.post("/file/upload_image")
async def upload_image(file: UploadFile = File(...),
                       task_prompt: str = "<MORE_DETAILED_CAPTION>",
                       ):

    # Kiểm tra file có hợp lệ
    tail = file.filename.split('.')[-1]
    if tail not in ['jpg', 'jpeg', 'png']:
        return JSONResponse(status_code=400, content={"message": "Invalid file type! The valid file will have extension: jpg, jpeg, png"})

    file_location = os.path.join(images_dir, file.filename)

    # Lưu file lên server
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load lại hình ảnh 
    image = Image.open(file_location)

    # Generate content   
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    
    return {"filename": file.filename,
            "content": parsed_answer}

@app.post("/file/upload_video")
async def upload_video(file: UploadFile = File(...),):
    
    tail = file.filename.split('.')[-1]
    if tail not in ['mp4',]:
        return JSONResponse(status_code=400, content={"message": "Invalid file type! The valid file will have extension: mp4"})
    
    file_location = os.path.join(images_dir, file.filename)

    # Lưu file lên server
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}