from __future__ import print_function
import datetime
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import webbrowser
import time
import pywhatkit as pwk
import speech_recognition as sr
import pyttsx3
import pytz
import subprocess

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
DAY_EXTENSIONS = ["rd", "nd", "th", "st"]


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(">>>")
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            speak("Say Something")
    return said.lower()


def authenticate_google():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)

    return service


def get_events(date, service):
    date = datetime.datetime.combine(date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(date, datetime.datetime.max.time())
    utc = pytz.UTC
    date = date.astimezone(utc)
    end_date = end_date.astimezone(utc)
    events_result = service.events().list(calendarId='primary', timeMin=date.isoformat(),
                                          timeMax=end_date.isoformat(), singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])

    if not events:
        speak('No events found on the day')
    else:
        speak(f"You have {len(events)} events on this day")
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            print(start, event['summary'])
            start_time = str(start.split("T")[1].split("-")[0])
            if int(start_time.split(":")[0]) < 12:
                start_time = start_time.split(":")[0] + start_time.split(":")[1] + " am"
            else:
                start_time = str(int(start_time.split(":")[0]) - 12) + start_time.split(":")[1] + " pm"
            speak(event["summary"] + " at " + start_time)


def get_date(text):
    text = text.lower()
    today = datetime.date.today()

    if text.count("today") > 0:
        return today
    day = -1
    day_of_week = -1
    month = -1
    year = today.year

    for word in text.split():
        if word in MONTHS:
            month = MONTHS.index(word) + 1
        elif word in DAYS:
            day_of_week = DAYS.index(word)
        elif word.isdigit():
            day = int(word)
        else:
            for ext in DAY_EXTENSIONS:
                found = word.find(ext)
                if found > 0:
                    try:
                        day = int(word[:found])
                    except:
                        pass
    if month < today.month and month != -1:
        year = year + 1
    if day < today.day and month == -1 and day != -1:
        month = today.month + 1
    if month == -1 and day == -1 and day_of_week != -1:
        current_day_of_week = today.weekday()
        dif = day_of_week - current_day_of_week
        if dif < 0:
            dif += 7
            if text.count("next") >= 1:
                dif += 7
        return today + datetime.timedelta(dif)
    if month == -1 or day == -1:
        return None
    return datetime.date(month=month, year=year, day=day)


def note(text):
    date = datetime.datetime.now()
    file_name = r"E:\pythonProject\SpeechRecognition\Notes\\" + str(date).replace(":", "-") + "-note.txt"
    with open(file_name, "w") as f:
        f.write(text)
    subprocess.Popen(["notepad.exe", file_name])


def there_exists(terms, text):
    for term in terms:
        if term in text:
            return True


def respond(f):
    while f == 0:
        speak("What do you want me to do?")
        text = get_audio()
        Calendar_str = ["what do i have", "do i have plans", "am i busy", "schedule"]
        if there_exists(Calendar_str, text):
            date = get_date(text)
            if date:
                get_events(get_date(text), service)
            else:
                speak("Please... Try Again")
        Note_str = ["make a note", "write this down", "remember this"]
        if there_exists(Note_str, text):
            speak("What would you like me to write down?")
            note_text = get_audio()
            note(note_text)
            speak("I've made a note of that.")
        if there_exists(['what is your name', 'what\'s your name', 'tell me your name'], text):
            speak("i am Bot the assisstant")
        if there_exists(["search for"], text) and 'youtube' not in text:
            search_term = text.split("for")[-1]
            url = f"https://google.com/search?q={search_term}"
            webbrowser.get().open(url)
            speak(f'Here is what i have found for {search_term} on google')
        if there_exists(["search on youtube"], text):
            search_term = text.split("for")[-1]
            url = f"https://www.youtube.com/results?search_query={search_term}"
            webbrowser.get().open(url)
            speak(f'Here is what i have found for {search_term} on youtube')
        if there_exists(["play"], text):
            search_term = text.split("play")[-1]
            pwk.playonyt(search_term)
            speak(f'Here is what i have found for {search_term} on youtube')
        Close_str = ["bye", "exit", "thank you"]
        for c in Close_str:
            if c in text:
                speak("Sad to know that you wanna exit... Thank You... and Bye......")
                return 1


Wake_str = "wake up"
service = authenticate_google()
speak("Say 'wake up' to trigger")
f = 0
while f == 0:
    text = get_audio()
    if text.count(Wake_str) > 0:
        speak("I am ready")
        f = respond(f)
