from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import sqlite3
from datetime import datetime

app = FastAPI()

# Create images folder
os.makedirs("images", exist_ok=True)
# Create videos folder
os.makedirs("videos", exist_ok=True)

# Create database and images table
def create_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            file_path TEXT,
            uploaded_time TEXT
        )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        file_path TEXT,
        uploaded_time TEXT
    )
""")

    conn.commit()
    conn.close()


# Run database creation
create_database()


# Home API
@app.get("/")
def home():
    return {
        "message": "SIH Backend is Working!"
    }


# Upload Image API
@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):

    # Save image
    file_path = f"images/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get upload time
    uploaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Store information in database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO images (filename, file_path, uploaded_time) VALUES (?, ?, ?)",
        (file.filename, file_path, uploaded_time)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Image stored successfully!",
        "filename": file.filename,
        "stored_at": file_path,
        "uploaded_time": uploaded_time
    }


# Get all stored images
@app.get("/images")
def get_images():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM images")

    rows = cursor.fetchall()

    conn.close()

    images_list = []

    for row in rows:
        images_list.append({
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "uploaded_time": row[3]
        })

    return images_list

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):

    file_path = f"videos/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    uploaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO videos (filename, file_path, uploaded_time) VALUES (?, ?, ?)",
        (file.filename, file_path, uploaded_time)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Video uploaded successfully!",
        "filename": file.filename,
        "stored_at": file_path,
        "uploaded_time": uploaded_time
    }

@app.get("/videos")
def get_videos():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM videos")

    rows = cursor.fetchall()

    conn.close()

    videos_list = []

    for row in rows:
        videos_list.append({
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "uploaded_time": row[3]
        })

    return videos_list


@app.get("/videos")
def get_videos():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM videos")

    rows = cursor.fetchall()

    conn.close()

    videos_list = []

    for row in rows:
        videos_list.append({
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "uploaded_time": row[3]
        })

    return videos_list

@app.get("/videos/{filename}")
def get_video(filename: str):
    file_path = f"videos/{filename}"

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")

    return {"error": "Video not found"}


