# app.py - Complete Flask Backend for YouTube Video Chat

# --- Imports ---
from flask import Flask, request, jsonify
from flask_cors import CORS # Required for cross-origin requests from React
from urllib.parse import urlparse, parse_qs
import math
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from chromadb.api.types import EmbeddingFunction # For ChromaDB custom embedding
import google.generativeai as genai
import os

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = "ApiKey"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the Gemini models
# chat_model is used for summarization, sectioning, and final answers
chat_model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-05-20")

# --- Define the Embedding Function as a class for ChromaDB ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom EmbeddingFunction for ChromaDB using Google Gemini's embedding-001 model.
    """
    def __call__(self, input_texts):
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        embeddings = []
        for text in input_texts:
            try:
                response = genai.embed_content(model="embedding-001", content=text)
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Error embedding text: '{text[:50]}...' - {e}")
                # Re-raise to ensure the error is propagated if a critical embedding fails
                raise e
        return embeddings

# --- Initialize ChromaDB Client and Collection (Global) ---
# This client and collection will be used across different Flask routes
client = chromadb.PersistentClient(path="./chroma_db")
video_collection = client.get_or_create_collection(
    name="youtube_video_transcripts",
    embedding_function=GeminiEmbeddingFunction()
)
print("ChromaDB collection initialized successfully.")


# --- Helper Functions  ---

def get_youtube_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    """
    parsed_url = urlparse(url)
    # Common YouTube hostnames 
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com', 'www.youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/') or parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    elif parsed_url.hostname == 'youtu.be': # Shortened URL
        return parsed_url.path[1:]
    return None

def format_timestamp(seconds):
    """Formats seconds into MM:SS or HH:MM:SS format."""
    seconds = math.floor(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def get_video_transcript(video_url):
    """Fetches the transcript of a YouTube video, returning both the joined text and the list of transcript segments with timestamps."""
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return None, None, "Could not extract video ID from the URL."

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript_list])
        return transcript_text, transcript_list, None
    except Exception as e:
        return None, None, f"Error fetching transcript: {e}"

def get_video_sections(transcript_text, video_url, gemini_model):
    """
    Uses Gemini to identify logical sections in the video transcript and
    generates hyperlinked timestamps.
    """
    if not transcript_text:
        return "No transcript available to create sections."

    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return "Could not determine video ID for timestamp links."

    prompt = f"""
    Based on the following video transcript, identify logical sections or topics discussed.
    For each section, provide:
    1. A concise title for the section.
    2. The approximate start time of that section in MM:SS format.

    Present the output as a numbered list. Example:
    1. Introduction - 00:00
    2. Main Topic - 01:23
    3. Conclusion - 05:45

    Video Transcript:
    {transcript_text}
    """

    try:
        response = gemini_model.generate_content(prompt)
        sections_raw_text = ""
        if response.parts and hasattr(response.parts[0], 'text'):
            sections_raw_text = response.parts[0].text
        elif hasattr(response, 'text'):
            sections_raw_text = response.text
        else:
            return "Could not generate sections: No content in the response."

        section_lines = sections_raw_text.strip().split('\n')
        markdown_sections = []
        for line in section_lines:
            parts = line.split(' - ')
            if len(parts) >= 2:
                title_part = " - ".join(parts[:-1]).strip()
                time_str = parts[-1].strip()
                
                try:
                    time_components = [int(x) for x in time_str.split(':')]
                    total_seconds = 0
                    if len(time_components) == 2:
                        total_seconds = time_components[0] * 60 + time_components[1]
                    elif len(time_components) == 3:
                        total_seconds = time_components[0] * 3600 + time_components[1] * 60 + time_components[2]

                    # Use the correct YouTube URL format for timestamps
                    hyperlink = f"https://www.youtube.com/watch?v=VIDEO_ID&t=START_SECONDSs${video_id}?t={total_seconds}s"
                    markdown_sections.append(f"{title_part} ([{time_str}]({hyperlink}))")
                except ValueError:
                    markdown_sections.append(line)
            else:
                markdown_sections.append(line)

        return "\n".join(markdown_sections)

    except Exception as e:
        return f"Error generating sections: {e}"

def summarize_text_with_gemini(text, gemini_model):
    """
    Summarizes the given text using the specified Gemini model.
    """
    if not text:
        return "No content provided to summarize."

    prompt = f"Please provide a concise summary:\n\n{text}"
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts and hasattr(response.parts[0], 'text'):
            return response.parts[0].text
        elif hasattr(response, 'text'):
            return response.text
        else:
            return "Could not generate summary: No content in the response."
    except Exception as e:
        return f"Error generating summary: {e}"

def create_transcript_chunks(transcript_list, chunk_size=10, overlap=2):
    """
    Creates overlapping text chunks from the raw transcript list,
    retaining start and end timestamps for each chunk.
    """
    chunks = []
    for i in range(0, len(transcript_list), chunk_size - overlap):
        chunk_segments = transcript_list[i : i + chunk_size]
        if not chunk_segments:
            continue

        chunk_text = " ".join([seg['text'] for seg in chunk_segments])
        start_time = chunk_segments[0]['start']
        end_time = chunk_segments[-1]['start'] + chunk_segments[-1]['duration']

        chunks.append({
            "text": chunk_text,
            "start_time": start_time,
            "end_time": end_time
        })
    return chunks

def embed_chunks_and_store_in_chromadb(video_id, transcript_chunks, chroma_collection):
    """
    Generates embeddings for each chunk and stores them in ChromaDB.
    """
    print(f"Embedding {len(transcript_chunks)} chunks for video ID: {video_id}...")
    documents = []
    metadatas = []
    ids = []
    embeddings_list = []

    for i, chunk in enumerate(transcript_chunks):
        try:
            embedding_response = genai.embed_content(model="embedding-001", content=chunk["text"])
            embedding = embedding_response['embedding']

            documents.append(chunk["text"])
            metadatas.append({
                "video_id": video_id,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "start_timestamp_formatted": format_timestamp(chunk["start_time"]),
                "end_timestamp_formatted": format_timestamp(chunk["end_time"])
            })
            ids.append(f"{video_id}_chunk_{i}")
            embeddings_list.append(embedding)
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            continue

    if documents:
        try:
            chroma_collection.add(
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(documents)} chunks to ChromaDB.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
    else:
        print("No documents to add to ChromaDB.")

def chat_with_video(user_query, video_id, chroma_collection, chat_model, top_k=3):
    """
    Finds relevant transcript chunks from ChromaDB and generates an answer using Gemini,
    including YouTube video clip URLs for the matched segments.
    """
    try:
        # Step 1: Embed the user query
        query_embedding_response = genai.embed_content(model="embedding-001", content=user_query)
        query_embedding = query_embedding_response['embedding']

        # Step 2: Query ChromaDB
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"video_id": video_id} # Ensure we only query for the current video
        )

        # Ensure results are not empty before proceeding
        if not results['documents'] or not results['documents'][0]:
            return "I couldn't find any relevant sections in the video to answer that question."

        matched_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]

        # Step 3: Generate full context from top matches
        context = ""
        citations = []
        for i, (chunk_text, meta) in enumerate(zip(matched_chunks, metadatas)):
            start = meta['start_time']
            end = meta['end_time']
            start_fmt = format_timestamp(start)
            end_fmt = format_timestamp(end)
            # Use the correct YouTube URL format for clips
            clip_url = f"https://www.youtube.com/watch?v=VIDEO_ID&t=START_SECONDSs${video_id}?start={int(start)}&end={int(end)}"
            context += f"Chunk {i+1} ({start_fmt} - {end_fmt}):\n{chunk_text}\n\n"
            citations.append(f"[Clip {i+1}: {start_fmt} - {end_fmt}]({clip_url})")

        # Step 4: Ask Gemini to answer the query using the matched transcript chunks
        prompt = f"""You are given transcript snippets of a YouTube video.
        Use them to answer the user query below. Include references to specific time ranges using the citations provided.

        Transcript Snippets:
        {context}

        User Query:
        {user_query}

        Citations:
        {chr(10).join(citations)}
        """
        response = chat_model.generate_content(prompt)

        # Combine answer with links to video clips
        answer_text = response.text if hasattr(response, 'text') else response.parts[0].text
        answer_with_clips = answer_text + "\n\n" + "\n".join(citations)
        return answer_with_clips

    except Exception as e:
        return f"Error during video chat: {e}"

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for your Flask app

# --- Flask Routes ---

@app.route('/')
def api_root():
    """
    This route serves as a simple confirmation that the Flask API is running.
    The main UI is handled by the React frontend on http://localhost:3000.
    """
    return jsonify({"message": "Flask API is running! Access the frontend at http://localhost:3000"})

@app.route('/process_video', methods=['POST'])
def process_video_route():
    """
    Handles the request to process a YouTube video.
    Extracts transcript, stores in ChromaDB, generates summary and sections.
    Returns JSON response to frontend.
    """
    data = request.get_json()
    video_url = data.get('video_url')
    
    if not video_url:
        return jsonify({"error": "Video URL is required"}), 400

    print(f"Received request to process video: {video_url}") # Debugging

    try:
        video_id = get_youtube_video_id(video_url)
        if not video_id:
            return jsonify({"error": "Could not extract video ID from the URL. Please check the format."}), 400

        transcript_text, transcript_list, error_message = get_video_transcript(video_url)

        if error_message:
            return jsonify({"error": f"Error fetching transcript: {error_message}"}), 500
        if not transcript_list:
            return jsonify({"error": "Could not retrieve transcript for this video. It might not have one or be unavailable."}), 500

        # Check if the video is already processed in ChromaDB
        existing_docs_count = len(video_collection.get(where={"video_id": video_id})['ids'])
        if existing_docs_count == 0:
            print(f"Processing new video ID: {video_id}. Embedding chunks...")
            chunks = create_transcript_chunks(transcript_list)
            embed_chunks_and_store_in_chromadb(video_id, chunks, video_collection)
        else:
            print(f"Video ID: {video_id} already processed and loaded into ChromaDB.")

        summary = summarize_text_with_gemini(transcript_text, chat_model)
        sections_markdown = get_video_sections(transcript_text, video_url, chat_model)

        return jsonify({
            "video_id": video_id,
            "summary": summary,
            "sections": sections_markdown,
            "message": "Video transcript processed! Ask me anything about it."
        })

    except Exception as e:
        print(f"An unexpected error occurred in /process_video: {e}") # Detailed error logging
        return jsonify({"error": f"An internal server error occurred during video processing: {str(e)}"}), 500


@app.route('/chat', methods=['POST'])
def chat_route():
    """
    Handles chat queries against the video transcript.
    Receives user query and video_id, uses Gemini to generate a response.
    Returns JSON response to frontend.
    """
    data = request.get_json()
    user_query = data.get('user_query')
    video_id = data.get('video_id')

    if not user_query or not video_id:
         return jsonify({"error": "Missing user_query or video_id"}), 400

    print(f"Received chat query for video {video_id}: '{user_query}'") # Debugging

    try:
        answer = chat_with_video(user_query, video_id, video_collection, chat_model)
        return jsonify({"response": answer})

    except Exception as e:
        print(f"An unexpected error occurred in /chat: {e}") # Detailed error logging
        return jsonify({"error": f"An internal server error occurred during chat: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)