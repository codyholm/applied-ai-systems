"""
Command line runner for the Music Recommender Simulation.

Runs the recommender against a set of named evaluation profiles and prints
the top recommendations with a short reason breakdown for each pick.
"""

from .recommender import UserProfile, load_songs, recommend_songs


user_profiles: list[tuple[str, UserProfile]] = [
    (
        "High-Energy Pop",
        UserProfile(
            favorite_genre="pop",
            favorite_mood="happy",
            target_energy=0.85,
            target_tempo_bpm=124.0,
            target_acousticness=0.15,
            target_valence=0.85,
            target_danceability=0.85,
        ),
    ),
    (
        "Chill Lofi",
        UserProfile(
            favorite_genre="lofi",
            favorite_mood="chill",
            target_energy=0.35,
            target_tempo_bpm=78.0,
            target_acousticness=0.80,
            target_valence=0.55,
            target_danceability=0.50,
        ),
    ),
    (
        "Deep Intense Rock",
        UserProfile(
            favorite_genre="rock",
            favorite_mood="intense",
            target_energy=0.90,
            target_tempo_bpm=145.0,
            target_acousticness=0.10,
            target_valence=0.45,
            target_danceability=0.65,
        ),
    ),
    (
        "Chill Rock",
        UserProfile(
            favorite_genre="rock",
            favorite_mood="chill",
            target_energy=0.25,
            target_tempo_bpm=75.0,
            target_acousticness=0.85,
            target_valence=0.50,
            target_danceability=0.30,
        ),
    ),
    (
        "Boundary Maximalist",
        UserProfile(
            favorite_genre="electronic",
            favorite_mood="intense",
            target_energy=1.0,
            target_tempo_bpm=200.0,
            target_acousticness=0.0,
            target_valence=1.0,
            target_danceability=1.0,
        ),
    ),
]


def main() -> None:
    songs = load_songs()
    print(f"Loaded songs: {len(songs)}")

    for profile_name, user in user_profiles:
        recommendations = recommend_songs(user, songs, k=5)

        print(f"\nProfile: {profile_name}")
        print("-" * (len(profile_name) + 9))

        for index, rec in enumerate(recommendations, start=1):
            print(f"{index}. {rec.song.title} by {rec.song.artist}")
            print(f"   Score:   {rec.score:.2f}")
            print(f"   Reasons: {', '.join(rec.reasons)}")
            print()


if __name__ == "__main__":
    main()
