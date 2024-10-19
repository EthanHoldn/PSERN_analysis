import whisper
import ssl
import certifi
import urllib.request
import requests
import pandas as pd
import time
import warnings
import joblib

# Load the classification model and vectorizer
classification_model = joblib.load('college_bio_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define relevant groups and channels for filtering audio calls
display_tags = ['Aubur PD Pri', 'Encaw PD Disp', 'FWPD Disp', 'Issaq/Snoq PD', 'KCSO North', 'KCSO NW', 'KCSO SE', 'KCSO SW', 'Kent PD Pri', 'Metro PD Pri', 'NC Pol 1', 'NC Pol 2', 'NC Pol 3', 'Red PD Disp', 'RetPD Pri', 'SPD Disp East', 'SPD Disp South', 'SPD Disp. North', 'SPD Disp. West'] 
groupings_tags = [] 

# Suppress the specific FP16 warning from Whisper (irrelevant for CPU usage)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Create a custom SSL context using certifi's certificates for secure connections
def setup_ssl_context():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context)))

# Load Whisper models
def load_whisper_models():
    return {
        'medium': whisper.load_model("medium.en"),
        'small': whisper.load_model("small.en"),
        'tiny': whisper.load_model("tiny.en")
    }

# Fetch live calls from the API
def fetch_live_calls(last_fetch_time):
    #
    # API URL and headers
    #
    response = requests.post(api_url, headers=headers, data=payload)
    return response.json()

# Filter and process calls from the fetched data
def process_calls(json_data, models, vectorizer, classification_model, calls_df, old_ttl):
    new_calls_df = pd.json_normalize(json_data["calls"])
    columns_to_keep = ['id', 'ts', 'systemId', 'filename', 'call_duration', 'call_tg', 'enc', 'hash', 'descr', 'display', 'grouping']
    filtered_calls_df = new_calls_df[columns_to_keep]

    for _, call in filtered_calls_df.iterrows():
        if call['grouping'] in groupings_tags or call['display'] in display_tags or "all" in groupings_tags or "all" in display_tags:
            print("\n\nNew call:")
            print(f"TTL: {time.time() - int(call['ts'])}")
            print(call['descr'] + " - " + call['display']) 
            old_ttl = time.time() - int(call['ts'])

            audio_url = f"https://calls.broadcastify.com/{call['hash']}/{call['systemId']}/{call['filename']}.{call['enc']}"
            audio_file = download_audio(audio_url, call['enc'])

            transcription_text = transcribe_audio(audio_file, old_ttl, models)
            prediction = classify_transcription(transcription_text, vectorizer, classification_model)

            calls_df = append_call_to_df(calls_df, call, transcription_text, audio_url, prediction)

    return calls_df, old_ttl

# Download audio file
def download_audio(audio_url, file_extension):
    audio_response = requests.get(audio_url)
    audio_file = f"tmp.{file_extension}"
    with open(audio_file, 'wb') as f:
        f.write(audio_response.content)
    return audio_file

# Transcribe the audio based on TTL
def transcribe_audio(audio_file, ttl, models):
    if ttl < 40:
        model = models['medium']
    elif ttl < 80:
        model = models['small']
    else:
        model = models['tiny']
    
    transcription_result = model.transcribe(audio_file)
    return transcription_result["text"]

# Classify the transcribed text
def classify_transcription(transcription_text, vectorizer, classification_model):
    transformed_text = vectorizer.transform([transcription_text])
    prediction = classification_model.predict_proba(transformed_text)
    return prediction[0][1]

# Append a call to the DataFrame
def append_call_to_df(calls_df, call, transcription_text, audio_url, prediction):
    new_row = {
        'id': call['id'],
        'timestamp': call['ts'],
        'system_id': call['systemId'],
        'filename': call['filename'],
        'call_duration': call['call_duration'],
        'call_tg': call['call_tg'],
        'encryption': call['enc'],
        'hash': call['hash'],
        'description': call['descr'],
        'display': call['display'],
        'grouping': call['grouping'],
        'transcription': transcription_text,
        'url': audio_url,
        'prediction': prediction
    }
    return pd.concat([calls_df, pd.DataFrame([new_row])], ignore_index=True)

# Main loop for fetching and processing calls
def main():
    setup_ssl_context()
    models = load_whisper_models()
    calls_df = pd.DataFrame(columns=[
        'id', 'timestamp', 'system_id', 'filename', 'call_duration', 'call_tg', 'encryption', 'hash', 'description', 
        'display', 'grouping', 'transcription'
    ])

    last_fetch_time = time.time() - 10
    old_ttl = 0
    try:
        while True:
            time.sleep(10)
            print("Fetching new calls...")
            json_data = fetch_live_calls(last_fetch_time)
            if "calls" not in json_data or not json_data["calls"]:
                print("No new calls.")
                continue
            calls_df, old_ttl = process_calls(json_data, models, vectorizer, classification_model, calls_df, old_ttl)
            last_fetch_time = time.time()
    except Exception as e:
        print("An error occurred:", e)
        calls_df.to_csv("calls.csv", index=False)

if __name__ == "__main__":
    main()
