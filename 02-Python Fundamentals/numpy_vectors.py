import numpy as np
dog = np.array([0.8,0.7,0.2])
puppy = np.array([0.78,0.72,0.25])
car = np.array([-0.3,0.1,0.9])

print("Dog",dog)
print("Puppy", puppy)
print("Car",car)

dog_puppy_dot = np.dot(dog,puppy)
print("Dog Puppy Dot Product:", dog_puppy_dot)

dog_magnitude = np.linalg.norm(dog)
puppy_magnitude = np.linalg.norm(puppy)

print("Dog Maginitude:", dog_magnitude)
print("Puppy Magnitude:", puppy_magnitude)

dog_puppy_similarity = dog_puppy_dot/(dog_magnitude * puppy_magnitude)
print("Dog Puppy Similarity:", dog_puppy_similarity)

car_magnitude = np.linalg.norm(car)
dog_car_dot = np.dot(dog,car)
dog_car_similarity = dog_car_dot/dog_magnitude*car_magnitude
print("Dog Car Similarity:", dog_car_similarity)

def cosine_similarity (vector_a, vector_b):

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    similarity = dot_product/ (magnitude_a * magnitude_b)

    return similarity

print("Cosine Similarity is:", cosine_similarity(dog,puppy))

cat = np.array([0.75,0.65,0.30])

dog = np.array([0.8,0.7,0.2])
puppy = np.array([0.78,0.72,0.25])
car = np.array([-0.3,0.1,0.9])
cat = np.array([0.75,0.65,0.30])

print("Dog vs puppy:", cosine_similarity(dog,puppy))
print("Dog vs Car:",cosine_similarity(dog, car))
print("Dog vs Cat:", cosine_similarity(dog, cat))

similarities = {"puppy":cosine_similarity(dog,puppy),
                "car":cosine_similarity(dog,car), 
                "cat":cosine_similarity(dog,cat)
                }
print("These are the similarities:",similarities)
most_similar = max(similarities, key=similarities.get)
print("most similar to dog:",most_similar)

scores = {"A":0.45, "B": 0.96, "C": 0.72}
print(max(scores,key=scores.get))
print(max(scores.values()))