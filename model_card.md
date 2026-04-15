# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

SongFinder 9000

---

## 2. Intended Use

SongFinder 9000 takes a listener's taste profile and ranks a small CSV
catalog of songs against it, returning the top 5 picks with a short
reason breakdown for each one. A taste profile is a favorite genre, a
favorite mood, and five numeric targets (energy, tempo, acousticness,
valence, danceability).

It assumes the listener can describe their taste as a single fixed
profile, and that the seven features in the catalog are enough to
capture what they want.

Not intended for anything where a bad recommendation would matter.
The catalog is 18 songs and the scoring is additive similarity, so
treating the output as a real listening suggestion would be a stretch.

---

## 3. How the Model Works

Every song has seven features: genre, mood, energy, tempo, acousticness,
valence (how positive the song sounds), and danceability. 
The listener gives the same seven back as a taste profile: a favorite genre, a favorite mood, and target values for the five numeric features.

Scoring compares one song to the profile, feature by feature. Genre
and mood are labels, so they either match or they don't. A match earns
a flat bonus. The five numeric features earn partial credit based on
how close the song is to the target. An exact match earns the full
weight for that feature. A bigger gap earns fewer points. All the
points get added together into one score.

Every song in the catalog gets scored the same way, the list gets
sorted from highest to lowest, and the top 5 come back with a short
note explaining where the points came from.

The starter scoring only looked at genre, mood, energy, tempo, and
acousticness. I added valence and danceability so the profile could
capture how a song feels (happy vs dark) and how it moves, not just
how active it is.

---

## 4. Data

The catalog is 18 songs loaded from `data/songs.csv`. The starter
shipped with 10 tracks and I added 8 more to broaden the coverage,
which brought in 3 new genres (electronic, acoustic, hip hop) and 2
new moods (upbeat, relaxed).

Each song has seven features: genre, mood, energy (0.0-1.0),
tempo in BPM, valence (0.0-1.0), danceability (0.0-1.0), and
acousticness (0.0-1.0).

Ten genres are represented: pop, lofi, indie pop, hip hop, electronic,
acoustic, rock, ambient, jazz, and synthwave. Pop and lofi have three
tracks each. Indie pop, hip hop, electronic, and acoustic have two
each. Rock, ambient, jazz, and synthwave have only one track each.
Moods cover happy, chill, intense, melancholy, focused, moody,
relaxed, and upbeat.

Plenty is missing. There's no classical, country, folk, metal, or
anything in a language other than English. No popularity or release
date. Everything in the catalog is a curated handful of songs, not a real library. 
A listener whose taste lives outside that handful gets a recommender that can't really represent them.

---

## 5. Strengths

The system does well on clean taste clusters. High-Energy Pop, Chill
Lofi, and Deep Intense Rock all returned top 5s that matched what I'd
expect for those listeners, with little overlap between them. When a
profile's label preferences and numeric targets point in the same
direction, the ranking is steady.

Every recommendation comes with a short reason breakdown showing which
features earned points and how many. That makes the picks easy to
audit. I can look at a surprising result and see exactly why it
scored where it did, which made the adversarial profiles useful
instead of mysterious.

---

## 6. Limitations and Bias

The clearest weakness I found is that the genre signal can be drowned
out entirely by the numeric features. Genre adds a flat `+1.5` bonus on
a label match, but the five numeric similarities can collectively
contribute up to `9.5` points. In the Chill Rock adversarial profile,
the only rock song in the catalog (Storm Runner) doesn't appear anywhere
in the top 5, because every other feature pulls toward the lofi /
acoustic cluster and the genre bonus can't close the gap.

This weakness is made worse by the dataset itself. The catalog has 18
songs across 10 genres, and rock, ambient, jazz, and synthwave each
have only one track. A profile asking for an underrepresented genre is
already unlikely to find a strong genre match, and when the one
available match also has the wrong numeric profile, the recommender
silently swaps it for a closer-feeling song from a different genre. The
listener gets something that scores well on the math but doesn't feel
like what they asked for.

---

## 7. Evaluation

I tested the recommender against five user profiles defined in
`src/main.py`:

- Three baseline profiles representing clean taste clusters: High-Energy
  Pop, Chill Lofi, and Deep Intense Rock.
- Two adversarial profiles designed to stress the scoring logic: Chill
  Rock (rock label paired with chill / low-energy / high-acoustic
  numerics) and Boundary Maximalist (electronic / intense at boundary
  values: energy `1.0`, tempo `200` BPM, acousticness `0.0`).

For each profile I ran `python -m src.main`, looked at the top 5
results, and checked whether they matched what I'd expect for a
listener with that taste.

The three baseline profiles returned what I'd expect. The adversarial
profiles produced two surprises worth recording:

- Chill Rock returned zero rock songs in the top 5. Library Rain (lofi)
  ranked first. The +1.5 genre bonus could not carry Storm Runner past
  songs that better matched the chill / low-energy / acoustic numerics.
- Boundary Maximalist returned Gym Hero (pop / intense) at #1, beating
  both electronic candidates in the catalog. The high valence and
  danceability targets pulled upbeat pop above the cooler-feeling
  electronic tracks even though electronic was the requested genre.

I also ran one weight-shift experiment: doubled `ENERGY_WEIGHT` from
`2.5` to `5.0` and halved `GENRE_WEIGHT` from `1.5` to `0.75`. Results
are documented in the Experiments You Tried section of the README. The
short version: energy is the dominant signal in the current recipe and
small changes there reorder rankings more than equivalent changes to
genre or acousticness. The output became different, not obviously more
accurate.

### Pair-by-pair comparisons

With five profiles (three baseline and two adversarial) there are ten
pairs to compare. One observation per pair:

1. High-Energy Pop and Chill Lofi produce disjoint top fives with zero
   overlap, the clearest sign the system separates obvious extremes.
2. High-Energy Pop and Deep Intense Rock share only Gym Hero (pop
   genre with intense mood acts as the crossover); otherwise they pull
   from different clusters because happy and intense are different
   vibes.
3. High-Energy Pop and Chill Rock are complete opposites on every
   numeric axis and share no songs. Chill Rock's rock label has no
   visible effect because the numerics pull toward the calm side.
4. High-Energy Pop and Boundary Maximalist overlap heavily (Gym Hero,
   City Afterglow, Summer Polaroid appear on both) because both ask
   for high energy, valence, and danceability, even though their genre
   targets differ.
5. Chill Lofi and Deep Intense Rock have zero overlap. Chill Lofi
   clusters around energy `0.30-0.40` and tempo 70-80 BPM, Deep Intense
   Rock around `0.80-0.90` and 100-150 BPM.
6. Chill Lofi and Chill Rock share three of the top 5 songs (Library
   Rain, Midnight Coding, Spacewalk Thoughts) even though one asks for
   lofi and the other for rock. Clearest proof that chill mood and
   low-energy numerics dominate while the genre label is effectively
   ignored.
7. Chill Lofi and Boundary Maximalist are opposites on every axis with
   zero overlap, confirming the system handles diametrically opposed
   profiles correctly.
8. Deep Intense Rock and Chill Rock share a genre label but have zero
   overlap because the mood and numerics disagree. Storm Runner tops
   one list and doesn't appear in the other's top 5.
9. Deep Intense Rock and Boundary Maximalist share four of the top 5
   (Gym Hero, Storm Runner, Backseat Freestyle, Broken Neon) in
   different orders because both want intense mood with high energy.
10. Chill Rock and Boundary Maximalist are mirror images with zero
    overlap. Chill Rock gets the calmest tracks, Boundary Maximalist
    gets the loudest. Each adversarial profile exposes a different
    weakness: genre getting ignored entirely for Chill Rock, and
    boundary inputs getting filled by whatever is numerically closest
    for Boundary Maximalist.

Why does Gym Hero keep showing up for both the High-Energy Pop listener
and the Boundary Maximalist listener? It's one of the loudest, most
danceable, most upbeat songs in the catalog, so anyone asking for
high-energy music with a positive feel gets it near the top regardless
of the genre they requested. Genre is a small bonus on top of a much
bigger numeric similarity score, and Gym Hero happens to be strong on
the features that matter most for an upbeat listener.

---

## 8. Future Work

A few things I'd change if I kept going:

- Grow the catalog, especially for underrepresented genres. One rock
  song means a rock listener has nothing real to pick from. More
  tracks per genre would give adversarial profiles somewhere to land.
- Loosen label matching so "indie pop" counts as pop and "focused"
  counts as chill. Exact-label matching breaks on adjacent vibes that
  any listener would hear as the same thing.
- Let the listener say how much genre matters to them. Right now genre
  is a flat bonus that the numerics can steamroll. Letting the profile
  weight genre higher would respect a deliberate genre choice instead
  of quietly swapping it for whatever is numerically closest.
