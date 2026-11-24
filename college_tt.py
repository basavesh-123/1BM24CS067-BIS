import random

# Subjects
subjects = ["Maths", "Physics", "Python", "DBMS",
            "Java", "English", "Computer Networks",
            "Operating Systems"]

days = ["Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday"]

periods = 3

# Generate timetable
def generate_timetable():
    timetable = {}

    for day in days:
        timetable[day] = random.sample(subjects, periods)

    return timetable

# Fitness function
def fitness(timetable):
    all_subjects = []

    for day in days:
        all_subjects.extend(timetable[day])

    conflicts = len(all_subjects) - len(set(all_subjects))

    return 1 / (1 + conflicts)

# Cuckoo Search
def cuckoo_search():
    nests = [generate_timetable() for _ in range(10)]

    for i in range(100):
        for j in range(10):
            new_nest = generate_timetable()
            k = random.randint(0, 9)

            if fitness(new_nest) > fitness(nests[k]):
                nests[k] = new_nest

        nests[random.randint(0, 9)] = generate_timetable()

    return max(nests, key=fitness)

# Display timetable
best = cuckoo_search()

print("\nCOLLEGE WEEKLY TIMETABLE")
print("=" * 40)

for day in days:
    print("\n", day)
    print("-" * 25)

    for period, subject in enumerate(best[day], 1):
        print("Period", period, ":", subject)

print("\nFitness:", fitness(best))
