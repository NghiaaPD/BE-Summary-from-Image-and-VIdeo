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

@app.post("/test_data")
async def generate_content(
    task_prompt: str = Form(...), 
    text_input: str = Form(...),    
    image: UploadFile = File(...)
):
    try:
        # Đường dẫn lưu file
        file_location = os.path.join(images_dir, image.filename)

        # Lưu file lên server
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Đọc kích thước của ảnh
        with Image.open(file_location) as img:
            image_size = img.size  # (width, height)

        # Tạo prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        # Giả sử bạn đã định nghĩa processor và model
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda:0")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1500,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Giả sử bạn đã định nghĩa processor với method post_process_generation
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image_size
        )

        return {"message": parsed_answer}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)