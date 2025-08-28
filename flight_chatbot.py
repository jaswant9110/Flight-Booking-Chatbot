import random
import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

nltk.download('punkt')

# Load datasets dynamically from paths
flight_data = pd.read_csv('flight_data.csv')
small_talk_data = pd.read_csv('small_talk.csv')

# Prepare intents for NLP-based processingf
intents = {
    "book_flight": ["book a flight", "reserve a flight", "flight booking"],
    "view_flights": ["view flights", "list flights", "show flights"],
    "small_talk": ["how are you", "tell me a joke", "what's up"],
    "farewell": ["exit", "quit", "goodbye"],
    "my_booking": ["what is my booking", "show my booking details", "booking info"],
    "update_booking": ["update my booking", "change my booking", "modify booking"],
    "cancel_booking": ["cancel my booking", "delete booking", "remove booking"],
    "my_name": ["what is my name", "do you know my name"],
    "bot_name": ["what is your name", "who are you"],
    "help": ["help", "what can you do", "commands"],
}

# Preprocess and vectorize intents for similarity matching
def preprocess_and_vectorize(phrases):
    tokenized = [" ".join(word_tokenize(phrase.lower())) for phrase in phrases]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(tokenized)
    return vectorizer, vectors

intent_phrases = []
intent_keys = []
for key, phrases in intents.items():
    intent_phrases.extend(phrases)
    intent_keys.extend([key] * len(phrases))

vectorizer, vectors = preprocess_and_vectorize(intent_phrases)

# Function to match user input to intents
def match_intent(user_input):
    tokenized_input = " ".join(word_tokenize(user_input.lower()))
    input_vector = vectorizer.transform([tokenized_input])
    similarity_scores = cosine_similarity(input_vector, vectors).flatten()
    max_index = np.argmax(similarity_scores)
    if similarity_scores[max_index] > 0.5:
        return intent_keys[max_index]
    return "unknown"

# Function to search flights
def search_flights(origin, destination, date, travel_class):
    try:
        date_obj = datetime.strptime(date, "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return False, "Invalid date format. Please use DD/MM/YYYY."

    flights = flight_data[
        (flight_data["origin"].str.lower() == origin.lower()) &
        (flight_data["destination"].str.lower() == destination.lower()) &
        (flight_data["departure_date"] == date_obj) &
        (flight_data["flight_class"].str.lower() == travel_class.lower())
    ]

    if not flights.empty:
        return True, flights.to_dict(orient="records")
    return False, "Sorry, no flights are available for the selected criteria. Try another. "

# Function to handle small talk
def handle_small_talk(user_input):
    response = small_talk_data[small_talk_data['question'].str.lower() == user_input.lower()]
    if not response.empty:
        return response.iloc[0]['response']
    return "I'm not sure how to respond to that, but I'm here to help with flights!"

# Function to display all available flights
def display_all_flights():
    print("Bot: Here is a list of all available flights:")
    for idx, flight in flight_data.iterrows():
        print(f"Flight {flight['flight_id']} from {flight['origin']} to {flight['destination']} on {flight['departure_date']} ({flight['flight_class']}) - ${flight['price']}")

# Function to display help commands
def display_help():
    print("Bot: Here are some things you can ask me to do:")
    for intent, phrases in intents.items():
        print(f"- {intent.replace('_', ' ').capitalize()}: {', '.join(phrases[:2])}...")

# Main chatbot functionality
def chatbot():
    print("Welcome to the Safar Flight Booking Assistant!")
    print("Bot: Hello! May I know your name?")
    user_name = input("You: ").strip()
    print(f"Bot: Nice to meet you, {user_name}! Can I have your phone number?")
    user_phone = input("You: ").strip()
    while not user_phone.isdigit():
        print("Bot: Please enter a valid phone number.")
        user_phone = input("You: ").strip()

    print("Bot: Thank you! You can ask me to book a flight, view all flights, or chat with me. Type 'exit' to quit.")

    bookings = []

    while True:
        user_input = input("You: ").strip()
        intent = match_intent(user_input)

        if intent == "farewell":
            print("Bot: Goodbye! Have a great day!")
            break

        elif intent == "help":
            display_help()

        elif intent == "book_flight":
            print("Bot: Sure, let's book a flight.")
            origin = input("Where are you departing from? ").strip()
            destination = input("Where are you traveling to? ").strip()
            date = input("What is your travel date? ").strip()
            travel_class = input("What class would you prefer (Economy/Business)? ").strip()

            available, flights = search_flights(origin, destination, date, travel_class)
            if available:
                print("Bot: Here are the available flights:")
                for idx, flight in enumerate(flights, start=1):
                    print(f"{idx}. Flight {flight['flight_id']} from {flight['origin']} to {flight['destination']} "
                          f"on {flight['departure_date']} ({flight['flight_class']}) - ${flight['price']}")
                while True:
                    user_selection = input("Select the flight by number or flight ID: ").strip()
                    try:
                        if user_selection.isdigit():
                            selected_flight_idx = int(user_selection) - 1
                            if 0 <= selected_flight_idx < len(flights):
                                selected_flight = flights[selected_flight_idx]
                                break
                            else:
                                print("Bot: Invalid selection. Please enter a valid flight number.")
                        else:
                            selected_flight = next(
                                flight for flight in flights if flight['flight_id'].lower() == user_selection.lower()
                            )
                            break
                    except (ValueError, StopIteration):
                        print("Bot: Invalid input. Please select a valid flight number or flight ID.")

                ref = random.randint(1000, 9999)
                booking_details = {
                    "name": user_name,
                    "phone": user_phone,
                    "origin": selected_flight["origin"],
                    "destination": selected_flight["destination"],
                    "date": selected_flight["departure_date"],
                    "class": selected_flight["flight_class"],
                    "price": selected_flight["price"],
                    "reference": ref,
                    "flight_id": selected_flight["flight_id"]
                }
                bookings.append(booking_details)
                print(f"Bot: Your booking is confirmed! Reference: {ref}.")
                print(f"Booking Details:\nName: {booking_details['name']}\nPhone: {booking_details['phone']}\n"
                      f"Flight: {booking_details['flight_id']}\nFrom: {booking_details['origin']} To: {booking_details['destination']}\n"
                      f"Date: {booking_details['date']} Class: {booking_details['class']}\nPrice: ${booking_details['price']}")
            else:
                print(f"Bot: {flights}")

        elif intent == "my_booking":
            if bookings:
                print("Bot: Here are your active bookings:")
                for booking in bookings:
                    print(f"Reference: {booking['reference']}\n"
                          f"Flight: {booking['flight_id']}\nFrom: {booking['origin']} To: {booking['destination']}\n"
                          f"Date: {booking['date']} Class: {booking['class']}\nPrice: ${booking['price']}\n")
            else:
                print("Bot: You have no active bookings.")

        elif intent == "my_name":
            print(f"Bot: Your name is {user_name}.")

        elif intent == "bot_name":
            print("Bot: My name is Flight Assistant! I'm here to help you.")

        elif intent == "update_booking":
            if bookings:
                print("Bot: Let's update your booking. What would you like to change?")
                print("Options: origin, destination, date, class (you can specify multiple, separated by commas)")
                update_fields = input("You: ").strip().lower().split(',')

                for booking in bookings:
                    for field in update_fields:
                        field = field.strip()
                        if field in booking:
                            new_value = input(f"Enter new {field}: ").strip()
                            if field == "date":
                                # Validate date format
                                try:
                                    datetime.strptime(new_value, "%d/%m/%Y")
                                except ValueError:
                                    print(f"Bot: Invalid date format for {field}. Please use DD/MM/YYYY.")
                                    continue
                            booking[field] = new_value
                            print(f"Bot: Your {field} has been updated to {new_value}.")
                        else:
                            print(f"Bot: Invalid field '{field}'. Skipping.")
            else:
                print("Bot: You have no active bookings to update.")

        elif intent == "cancel_booking":
            if bookings:
                print("Bot: Are you sure you want to cancel all bookings? (yes/no)")
                confirmation = input("You: ").strip().lower()
                if confirmation == "yes":
                    bookings.clear()
                    print("Bot: All your bookings have been cancelled.")
                else:
                    print("Bot: Cancellation aborted.")
            else:
                print("Bot: You have no active bookings to cancel.")

        elif intent == "view_flights":
            display_all_flights()

        elif intent == "small_talk":
            response = handle_small_talk(user_input)
            print(f"Bot: {response}")

        else:
            print("Bot: I'm sorry, I didn't understand that. Can you please try again?")

if __name__ == "__main__":
    chatbot()
