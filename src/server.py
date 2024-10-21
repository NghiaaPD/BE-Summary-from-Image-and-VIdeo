from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import shutil
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import os
from video import summarize_video_content
from PIL import Image

# Load model
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True).eval().cuda()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True)
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

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

    inputs = florence_processor(text=task_prompt, images=image, return_tensors="pt").to("cuda:0")
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    
    return {"filename": file.filename,
            "content": parsed_answer}

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
        inputs = florence_processor(text=prompt, images=img, return_tensors="pt").to("cuda:0")

        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1500,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Giả sử bạn đã định nghĩa processor với method post_process_generation
        parsed_answer = florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image_size
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error: {e}"})

    return {"message": parsed_answer}


@app.post("/file/upload_video")
async def upload_video(file: UploadFile = File(...),):
    
    tail = file.filename.split('.')[-1]
    if tail not in ['mp4',]:
        return JSONResponse(status_code=400, content={"message": "Invalid file type! The valid file will have extension: mp4"})
    
    file_location = os.path.join(images_dir, file.filename)

    # Lưu file lên server
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    summary = summarize_video_content(file_location, 
                                      frame_rate=10, 
                                      florence_processor=florence_processor, 
                                      florence_model=florence_model, 
                                      bart_model=bart_model, 
                                      bart_tokenizer=bart_tokenizer)


    return {"filename": file.filename,
            "content": summary}