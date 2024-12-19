import cv2
import mediapipe as mp
import random   

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to detect Rock, Paper, Scissors gesture
def detect_rps_gesture(landmarks):
    thumb_is_open = landmarks[4][0] > landmarks[3][0]
    index_is_open = landmarks[8][1] < landmarks[6][1]
    middle_is_open = landmarks[12][1] < landmarks[10][1]
    ring_is_open = landmarks[16][1] < landmarks[14][1]
    pinky_is_open = landmarks[20][1] < landmarks[18][1]

    if not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Rock"
    elif thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Paper"
    elif index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "Scissors"
    else:
        return "Unknown Gesture"

# Function to get computer's move
def get_computer_move():
    return random.choice(["Rock", "Paper", "Scissors"])

# Function to determine the winner
def determine_winner(player_move, computer_move):
    if player_move == computer_move:
        return "Draw"
    elif (player_move == "Rock" and computer_move == "Scissors") or \
         (player_move == "Paper" and computer_move == "Rock") or \
         (player_move == "Scissors" and computer_move == "Paper"):
        return "Player"
    else:
        return "Computer"

# Set the score to win the game
winning_score = int(5)
player_score = 0
computer_score = 0

while player_score < winning_score and computer_score < winning_score:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image to find hands
    result = hands.process(img_rgb)
    
    player_move = "Unknown Gesture"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * img.shape[1])
                lmy = int(lm.y * img.shape[0])
                landmarks.append((lmx, lmy))
            
            # Detect and display RPS gesture
            if landmarks:
                player_move = detect_rps_gesture(landmarks)
                if player_move != "Unknown Gesture":
                    break  # Use the gesture from the first detected hand
    
    if player_move != "Unknown Gesture":
        computer_move = get_computer_move()
        winner = determine_winner(player_move, computer_move)
        
        if winner == "Player":
            player_score += 1
        elif winner == "Computer":
            computer_score += 1

        # Display moves and scores
        cv2.putText(img, f'Player: {player_move}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Computer: {computer_move}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Winner: {winner}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Player Score: {player_score}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f'Computer Score: {computer_score}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Wait for a short time before the next move
        cv2.imshow("Rock Paper Scissors Recognition", img)
        cv2.waitKey(5000)
    else:
        cv2.putText(img, 'Show your move!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow("Rock Paper Scissors Recognition", img)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

if player_score > computer_score:
    print("You won the game!")
else:
    print("Computer won the game!")
