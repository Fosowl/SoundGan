#!/usr/bin python3

import os
from openai import OpenAI
import pandas as pd
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

MIN_DURATION = 60
MAX_DURATION = 1800

search_queries = {
    "bird": {
      "english": ["bird sound", "bird singing", "bird chirping"],
    }
}

# convert youtube duration to seconds
def convert_youtube_duration(duration):
    reptms = re.compile(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$')
    hours, minutes, seconds, totalseconds = 0, 0, 0, 0

    if reptms.match(duration):
        matches = reptms.match(duration)
        if matches.group(1):
            hours = int(matches.group(1))
        if matches.group(2):
            minutes = int(matches.group(2))
        if matches.group(3):
            seconds = int(matches.group(3))
        totalseconds = hours * 3600 + minutes * 60 + seconds
    return totalseconds


def gpt_call(prompt):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You have to classify youtube video."},
            {
                "role": "user",
                "content": prompt 
            }
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response


def llm_check_title(title, query):
    prompt = f"""you must tell based on a youtube video title if it is very likely to contain {query} sound, answer by YES or NO only.
    Title: {title}
    Does this video contain {query} sound? yes or no.
    Answer:
    """
    generated = gpt_call(prompt)
    if "yes" in generated.lower():
        return True
    return False

# pick video from youtube search results
def choose_video(youtube, result, language, count, total, class_name):
    choosen = []
    for item in result['items']:
        count += 1
        descr = item['snippet']['description'].lower()
        title = item['snippet']['title'].lower()
        if not 'videoId' in item['id']:
            continue
        if not item['id']['kind'] == 'youtube#video':
            continue
        # TODO add exclusion system here
        if llm_check_title(title, class_name) == False:
            continue
        video_id = item['id']['videoId']
        video_details = youtube.videos().list(
            part='contentDetails',
            id=video_id
        ).execute()
        for video in video_details['items']:
            if not 'contentDetails' in video:
                continue
            duration_iso = video['contentDetails']['duration']
            duration_s = convert_youtube_duration(duration_iso)
            if duration_s > MIN_DURATION or duration_s < MAX_DURATION:
                print("Saving video", item['snippet']['title'])
                choosen.append(item)
    return choosen, count
  
# get youtube search results for a query
def get_youtube_results(youtube, query, language, max_result, class_name, config):
    next_page_token = None
    all_results = []
    count = 0
    while len(all_results) < config["RESULT_PER_QUERY"]:
        search_exlusion = ""
        result = youtube.search().list(
          q=query + search_exlusion,
          part='id,snippet',
          maxResults=config["VIDEO_PER_PAGE"],
          pageToken=next_page_token
        ).execute()
        print(f"\nSEARCH: << {query} >> - got {len(result['items'])} results.")
        choices, count = choose_video(youtube, result, language, count, max_result, class_name)
        all_results.extend(choices)
        next_page_token = result.get('nextPageToken')
        if not next_page_token:
            break
    return all_results

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")

def create_file_if_not_exists(file_path):
    try:
        with open(file_path, 'x') as f:
            f.write(",title,url,id,description")
            print(f"File {file_path} created.")
    except FileExistsError:
        pass
    except Exception as e:
        print(e)

def save_to_csv(data, csv_path):
    frame = pd.DataFrame(data=data)
    frame.drop_duplicates(subset="title", keep="first")
    with open(csv_path, 'a') as f:
        print(f"Saving to {csv_path}")
        frame.to_csv(f, header=True)
  
# save choosed videos to csv
def save_choices(choices, path):
    print(f"Saving {len(choices)} video in {path}")
    titles = []
    urls = []
    ids = []
    descriptions = []

    for video in choices:
        titles.append(video['snippet']['title'])
        descriptions.append(video['snippet']['description'])
        id = video['id']['videoId']
        ids.append(id)
        urls.append(f"https://www.youtube.com/watch?v={id}")
        print(f"Video saved in save buffer: {video['snippet']['title']}")
    data = {'title': titles, 'url': urls, 'id': ids, 'description': descriptions}
    save_to_csv(data, path)

# check if video is already saved
def already_saved(x, choices):
    for c in choices:
        if c['etag'] == x['etag']:
            return True
    return False

# do iterative youtube search on youtube and choose videos
def youtube_search(query, csv_file, config, dev_key):
    ytsesh = build(config["YOUTUBE_API_SERVICE_NAME"], config["YOUTUBE_API_VERSION"],
      developerKey=dev_key)
    total_count = 0
    total_estimate = len(search_queries) * 4 * config["RESULT_PER_QUERY"]
    choices = []
    create_folder_if_not_exists(config["CSV_FOLDER_PATH"])
    for language in search_queries[query]:
        print(f"\n--\n Switching to {language}...\n--\n")
        for search_term in search_queries[query][language]:
            print(f"\n--\nSearching for {search_term}...\n--\n")
            results = get_youtube_results(ytsesh, search_term, language, total_estimate, query, config)
            for search_result in results:
                if search_result['id']['kind'] == 'youtube#video':
                    if already_saved(search_result, choices) == False:
                        choices.append(search_result)
                        print("Video added:", search_result['snippet']['title'])
                total_count += 1
            print(f"\nSaving {len(choices)} video in {csv_file}")
            print(f"Total of {total_count} video analysed")
            save_choices(choices, csv_file)
            if total_count >= config["MAX_VIDEO_COUNT"]:
                break
    print(f"Total of {total_count} video analysed")
    print(f"Total of {len(choices)} video saved")

def scrawler(config, class_name):
    csv_file = f"{config['CSV_FOLDER_PATH']}/{class_name}.csv"
    create_file_if_not_exists(csv_file)
    try:
        key = os.getenv("YOUTUBE_API_KEY")
    except:
        raise Exception("Please set the YOUTUBE_API_KEY environment variable.")
    try:
        youtube_search(class_name, csv_file, config, dev_key=key)
    except HttpError as e:
        print(e)

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