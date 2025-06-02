import pandas as pd
import random
from faker import Faker
from datetime import datetime

fake = Faker()

# Parameters
NUM_USERS = 300
NUM_MOVIES = 200
NUM_INTERACTIONS = 10000

# Helper Data
genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi', 'Horror', 'Animation']
devices = ['Mobile', 'Tablet', 'Smart TV', 'Laptop', 'Desktop']
events = ['play', 'pause', 'stop', 'complete']

# Generate Realistic Movie Titles
def generate_movie_title():
    adjectives = ['Silent', 'Final', 'Broken', 'Eternal', 'Dark', 'Bright', 'Hidden', 'Lost', 'Last', 'Infinite']
    nouns = ['Hope', 'Dream', 'Shadow', 'Edge', 'Legacy', 'Memory', 'Truth', 'Storm', 'Hour', 'Whisper']
    return f"{random.choice(adjectives)} {random.choice(nouns)}"

# Generate Users
def generate_users(n):
    genders = ['Male', 'Female', 'Other']
    return [{
        'user_id': i,
        'user_name': fake.name(),
        'age': random.randint(16, 60),
        'gender': random.choice(genders),
        'location': fake.city()
    } for i in range(n)]

# Generate Movies
def generate_movies(n):
    return [{
        'movie_id': i,
        'title': generate_movie_title(),
        'genre': random.choice(genres),
        'release_year': random.randint(1995, 2024),
        'duration_min': random.randint(60, 180)
    } for i in range(n)]

# Generate OTT Interactions
def generate_interactions(n, num_users, num_movies):
    interactions = []
    for _ in range(n):
        user_id = random.randint(0, num_users - 1)
        movie_id = random.randint(0, num_movies - 1)
        event = random.choices(events, weights=[0.5, 0.2, 0.1, 0.2])[0]
        rating = round(random.uniform(1, 5), 1) if event == 'complete' else None
        watch_time = random.randint(1, 120) if event != 'complete' else random.randint(60, 180)
        timestamp = fake.date_time_between(start_date='-2y', end_date='now')
        device = random.choice(devices)

        interactions.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'event': event,
            'watch_time_min': watch_time,
            'device': device,
            'timestamp': timestamp,
            'rating': rating
        })
    return interactions

# Generate data
users_df = pd.DataFrame(generate_users(NUM_USERS))
movies_df = pd.DataFrame(generate_movies(NUM_MOVIES))
movies_df['movie_name'] = movies_df['title']  # Add explicit movie_name column

interactions_df = pd.DataFrame(generate_interactions(NUM_INTERACTIONS, NUM_USERS, NUM_MOVIES))

# Merge all into one full dataset
full_df = interactions_df.merge(users_df, on='user_id').merge(movies_df, on='movie_id')

# Save to CSV
full_df.to_csv("ott_full_dataset.csv", index=False)

print("✅ OTT full dataset with `movie_name` column generated as 'ott_full_dataset.csv'")
print("📊 Sample:")
print(full_df[['user_name', 'movie_name', 'event', 'watch_time_min', 'rating']].sample(5))
