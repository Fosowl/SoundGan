import os
import logging
import pandas as pd
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MIN_DURATION = 60
MAX_DURATION = 1800

# Convert YouTube duration to seconds
def convert_youtube_duration(duration):
    pattern = re.compile(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$')
    hours, minutes, seconds = 0, 0, 0

    match = pattern.match(duration)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

# GPT call for title classification
def gpt_call(prompt):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You have to classify YouTube video."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    return response

# Check if title matches the query using LLM
def llm_check_title(title, query):
    prompt = f"""You must tell based on a YouTube video title if it is very likely to contain {query} sound. Answer by YES or NO only.
    Title: {title}
    Does this video contain {query} sound? Yes or No.
    Answer:"""
    generated = gpt_call(prompt)
    result = "yes" in generated.lower()
    return result

# Choose videos from search results
def choose_video(youtube, result, count, total, class_name):
    logging.info(f"Choosing videos for class: {class_name}")
    chosen = []
    for item in result['items']:
        count += 1
        title = item['snippet']['title'].lower()

        if 'videoId' not in item['id'] or item['id']['kind'] != 'youtube#video':
            logging.debug(f"Skipping non video item or missing videoId: {item}")
            continue

        if not llm_check_title(title, class_name):
            logging.debug(f"Title rejected by LLM: {title}")
            continue

        video_id = item['id']['videoId']
        video_details = youtube.videos().list(
            part='contentDetails',
            id=video_id
        ).execute()

        for video in video_details['items']:
            duration_iso = video['contentDetails']['duration']
            duration_s = convert_youtube_duration(duration_iso)

            if MIN_DURATION <= duration_s <= MAX_DURATION:
                logging.info(f"Saving video: {item['snippet']['title']}")
                chosen.append(item)
    return chosen, count

# Get YouTube search results for a query
def get_youtube_results(youtube, query, max_result, class_name, config):
    logging.info(f"Fetching YouTube results for query: {query}")
    next_page_token = None
    all_results = []
    count = 0

    while len(all_results) < config["RESULT_PER_QUERY"]:
        search_exclusion = ""
        result = youtube.search().list(
            q=query + search_exclusion,
            part='id,snippet',
            maxResults=config["VIDEO_PER_PAGE"],
            pageToken=next_page_token
        ).execute()

        logging.info(f"SEARCH: << {query} >> - got {len(result['items'])} results.")
        choices, count = choose_video(youtube, result, count, max_result, class_name)
        all_results.extend(choices)
        next_page_token = result.get('nextPageToken')

        if not next_page_token:
            break

    return all_results

# Create folder if it does not exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Create file if it does not exist
def create_file_if_not_exists(file_path):
    try:
        with open(file_path, 'x') as f:
            f.write("title,url,id,description\n")
    except FileExistsError:
        logging.debug(f"File already exists: {file_path}")
    except Exception as e:
        logging.error(f"Error creating file: {e}")

# Save data to CSV
def save_to_csv(data, csv_path):
    logging.info(f"Saving data to CSV at: {csv_path}")
    frame = pd.DataFrame(data=data)
    frame.drop_duplicates(subset="title", keep="first", inplace=True)
    with open(csv_path, 'a') as f:
        frame.to_csv(f, header=f.tell()==0, index=False)

# Save chosen videos to CSV
def save_choices(choices, path):
    logging.info(f"Saving {len(choices)} videos to {path}")

    data = {
        'title': [video['snippet']['title'] for video in choices],
        'url': [f"https://www.youtube.com/watch?v={video['id']['videoId']}" for video in choices],
        'id': [video['id']['videoId'] for video in choices],
        'description': [video['snippet']['description'] for video in choices]
    }
    save_to_csv(data, path)

# Check if a video is already saved
def already_saved(video, choices):
    result = any(c['etag'] == video['etag'] for c in choices)
    return result

# Perform iterative YouTube search
def youtube_search(query, csv_file, config, dev_key):
    logging.info(f"Starting YouTube search for query: {query}")
    youtube = build(
        config["YOUTUBE_API_SERVICE_NAME"],
        config["YOUTUBE_API_VERSION"],
        developerKey=dev_key
    )

    total_count = 0
    choices = []
    create_folder_if_not_exists(config["CSV_FOLDER_PATH"])
    search_queries = [f"{query} sound", f"{query} noise", f"{query} clip", f"{query} recording", f"{query} ambience"]

    for search_term in search_queries:
        logging.info(f"Searching for {search_term}...")

        results = get_youtube_results(youtube, search_term, total_count, query, config)

        for result in results:
            if result['id']['kind'] == 'youtube#video' and not already_saved(result, choices):
                choices.append(result)
                logging.info(f"Video added: {result['snippet']['title']}")
            total_count += 1

        save_choices(choices, csv_file)
        if total_count >= config["MAX_VIDEO_COUNT"]:
            break

    logging.info(f"Total analyzed: {total_count} videos. Total saved: {len(choices)} videos.")

# Main scrawler function
def scrawler(config, class_name):
    csv_file = f"{config['CSV_FOLDER_PATH']}/{class_name}.csv"
    create_file_if_not_exists(csv_file)

    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise Exception("Please set the YOUTUBE_API_KEY environment variable.")

    try:
        youtube_search(class_name, csv_file, config, dev_key=key)
    except HttpError as e:
        logging.error(f"HTTP Error during YouTube search: {e}")

if __name__ == "__main__":
    config = {
        "CSV_FOLDER_PATH": "../prepared_data/csv/",
        "YOUTUBE_API_SERVICE_NAME": "youtube",
        "YOUTUBE_API_VERSION": "v3",
        "VIDEO_PER_PAGE": 5,
        "MAX_VIDEO_COUNT": 100,
        "RESULT_PER_QUERY": 10
    }
    scrawler(config, "bird")
