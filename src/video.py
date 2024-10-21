import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torchvision.transforms as transforms
import os


# Load model
# florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True).eval().cuda()
# florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True)
# bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Trích xuất frames từ video
def extract_frames(video_path, frame_interval=10):
    video = cv2.VideoCapture(video_path)
    count_frames = 0

    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            break
        
        if count_frames % frame_interval == 0:
            # cv2.imshow('Output: ', frame)
            cv2.imwrite(f'frames/frame_{count_frames}.jpg', frame)
        count_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()


# Mô tả từng khung hình bằng model Florence-2
def describe_frames(frames, florence_processor, florence_model):
    task_prompt = "<MORE_DETAILED_CAPTION>"
    frame_descriptions = []
    for frame in frames:
        image = Image.open(os.path.join('frames', frame))

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
        frame_descriptions.append(parsed_answer)
    
    return frame_descriptions


# Tổng hợp thông tin bằng BART 
def summarize_descriptions(descriptions, bart_model, bart_tokenizer):
    # Ghép nối các mô tả thành một đoạn văn bản
    combined_text = " ".join([desc["<MORE_DETAILED_CAPTION>"] for desc in descriptions])

    # Tokenizer và tạo input cho model BART
    inputs = bart_tokenizer.encode("summarize: " + combined_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Dự đoán tóm tắt
    summary_ids = bart_model.generate(inputs, max_length=200, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


# PIPELINE
def summarize_video_content(video_path, frame_rate=1, florence_processor=None, florence_model=None, bart_model=None, bart_tokenizer=None):
    # Xoá folder frame hiện tại
    if os.path.exists('frames'):
        for frame in os.listdir('frames'):
            os.remove(os.path.join('frames', frame))
    
    extract_frames(video_path, frame_rate)
    frames = os.listdir('frames')
    descriptions = describe_frames(frames, florence_processor, florence_model)
    summary = summarize_descriptions(descriptions, bart_model, bart_tokenizer)
    return summary


