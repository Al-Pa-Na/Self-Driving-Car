def check_proximity(tracked_objects, frame_width, frame_height):
    danger_zone_x1 = int(frame_width * 0.4)
    danger_zone_x2 = int(frame_width * 0.6)
    danger_zone_y1 = int(frame_height * 0.5)
    danger_zone_y2 = frame_height

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2

        if (danger_zone_x1 <= obj_center_x <= danger_zone_x2) and (danger_zone_y1 <= obj_center_y <= danger_zone_y2):
            return "Object Ahead"

    return "Clear"
