import mediapipe as mp 
import numpy as np 
import cv2 
 
cap = cv2.VideoCapture(0)

name = input("Enter the name of the data : ")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)


		X.append(lst)
		data_size = data_size+1



	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27 or data_size>99:
		cv2.destroyAllWindows()
		cap.release()
		break



#  how to save

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)


# import json
# import time

# # Path to your local storage file
# LOCAL_STORAGE_PATH = 'your_local_storage_file.json'

# # Example function to retrieve the temperature
# def get_temperature():
#     # Add your own code here to get the temperature
#     return 25.0 # example value

# # Example function to retrieve the humidity
# def get_humidity():
#     # Add your own code here to get the humidity
#     return 65.0 # example value

# # Main function to start data collection
# def start_data_collection():
#     # Define your data collection time in seconds
#     collection_time = 60 * 60 * 24 # example: 24 hours

#     # Load existing data from local storage file
#     try:
#         with open(LOCAL_STORAGE_PATH, 'r') as f:
#             data = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         data = []

#     # Start data collection loop
#     start_time = time.time()
#     while time.time() - start_time < collection_time:
#         # Get the temperature and humidity
#         temperature = get_temperature()
#         humidity = get_humidity()

#         # Store the collected data
#         data.append({
#             'timestamp': time.time(),
#             'temperature': temperature,
#             'humidity': humidity
#         })

#         # Wait for some time before collecting the next data
#         time.sleep(60) # example: wait for 1 minute before collecting the next data

#     # Save the collected data to the local storage file
#     with open(LOCAL_STORAGE_PATH, 'w') as f:
#         json.dump(data, f)

# if __name__ == '__main__':
#     start_data_collection()