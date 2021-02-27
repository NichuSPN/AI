import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import pywhatkit as pk
import webbrowser
import os
import smtplib

userName = "Nishaanth"

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + userName)
    elif hour >= 12 and hour < 17:
        speak("Good Afternoon " + userName)
    else:
        speak("Good Evening " + userName)


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"user said: {query}\n")

    except Exception as e:
        speak("Say that again please")
        query = None
    return query.lower()


def recogninze(query):
    if 'wikipedia' in query:
        speak('Searching wikipedia')
        query = query.replace('wikipedia', "")
        try:
            results = wikipedia.summary(query, sentences=2)
        except Exception as e:
            results = "Sorry could not find the page"
        speak(results)
    elif 'open youtube' in query:
        webbrowser.open("youtube.com")
    elif 'open reddit' in query:
        webbrowser.open('reddit.com')
    elif 'open google' in query:
        webbrowser.open("google.com")
    elif 'play music' in query:
        pk.playonyt("Tamil melody hit songs")
    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The time is {strTime}")
    elif 'search for ' in query and 'youtube' in query:
        query = query.replace('youtube', "")
        query = query.replace('search for', "")
        pk.playonyt(query)


speak("Initializing Jarvis...")
engine.setProperty('voice', voices[0].id)
wishMe()
q = takeCommand()
recogninze(q)
