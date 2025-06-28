# Decide steering action based on lane direction
def decide_steering_action(direction):
    if direction == "Turn Left":
        return "Steer Left"
    elif direction == "Turn Right":
        return "Steer Right"
    else:
        return "Maintain"
