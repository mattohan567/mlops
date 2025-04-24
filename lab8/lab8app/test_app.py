import requests
import json

# API endpoint (using the new port)
url = "http://localhost:8001/predict"

# Test with a single set of features
single_features = [0.1, 0.5, 0.2, 0.3, 0.8, 0.1, 0.5, 0.2, 0.3, 0.8]  # Example feature values

response = requests.post(
    url,
    json={"features": single_features}
)

print("Single Features Test:")
print(f"Features: {single_features}")
print(f"Response status code: {response.status_code}")
print(f"Prediction result: {json.dumps(response.json(), indent=2)}")
print("\n" + "-"*50 + "\n")

# Test with another set of features
second_features = [0.8, 0.2, 0.7, 0.1, 0.5, 0.8, 0.2, 0.7, 0.1, 0.5]  # Different example feature values
response = requests.post(
    url,
    json={"features": second_features}
)

print("Second Features Test:")
print(f"Features: {second_features}")
print(f"Response status code: {response.status_code}")
print(f"Prediction result: {json.dumps(response.json(), indent=2)}") 

print("\n" + "-"*50 + "\n")

# Test with all ones
all_ones = [1.0] * 10  # Create list of 10 ones
response = requests.post(
    url,
    json={"features": all_ones}
)

print("All Ones Test:")
print(f"Features: {all_ones}")
print(f"Response status code: {response.status_code}")
print(f"Prediction result: {json.dumps(response.json(), indent=2)}")
print("\n" + "-"*50 + "\n")

# Test with all twos
all_twos = [0] * 10  # Create list of 10 twos
response = requests.post(
    url,
    json={"features": all_twos}
)

print("All Twos Test:")
print(f"Features: {all_twos}")
print(f"Response status code: {response.status_code}")
print(f"Prediction result: {json.dumps(response.json(), indent=2)}")
