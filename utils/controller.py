# Decide steering action based on lane direction
def decide_steering_action(direction):
    if direction == "Turn Left":
        return -25
    elif direction == "Turn Right":
        return 25
    else:
        return 0
