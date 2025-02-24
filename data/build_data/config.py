def get_train_data(
        sup_image_size, sup_img_bbox, que_image_size, que_img_bbox, cls_name
):
    instruction = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"You are a highly skilled data annotator, capable of analyzing feature vectors to identify patterns and provide accurate annotations. "
        f"Your current task is to annotate objects within the class '{cls_name}' in an image by identifying their bounding boxes. "
        f"Specifically, you need to output the coordinates of a rectangle that encloses each object of this class. "
        f"The coordinates should be arranged in a clockwise direction and represented as a tuple in the format: (c1, c2), "
        f"where c1 and c2 are the coordinates for the top-left and bottom-right points of the rectangle, respectively. "
        f"The format of the coordinates should be [x_min, y_min, x_max, y_max]. "
        f"Ensure that all coordinate values are within the boundaries of the image size."
        f"\nThe input include the image size and the image in a structured JSON format like so:"
        f"\n"
        f"\nInput:"
        f"\n{{"
        f"\n  \"Image Size\": {sup_image_size},"
        f"\n  \"Image\": <visual_sup>"
        f"\n}}"
        f"\nwhere the position of <visual_sup> represents the image."

        f"\n\nYour output should include the bounding box coordinates and their corresponding class in a structured JSON format like so:"
        f"\n{{"
        f"\n  \"class\": \"{cls_name}\","
        f"\n  \"bounding_box\": {sup_img_bbox}"
        f"\n}}"
        f"<|eot_id|>"
    )

    input = (
        f"<|start_header_id|>user<|end_header_id|>"
        f'{{'
        f'\n  "Image Size":{que_image_size},'
        f'\n  "Image": <visual_que>'
        f'\n}}'
        f'<|eot_id|>'
        f'<|start_header_id|>assistant<|end_header_id|>'
    )

    output = ( 
        f"{{"
        f"\n  \"class\": \"{cls_name}\","
        f"\n  \"bounding_box\": {que_img_bbox}"
        f"\n}}"
        f"<|eot_id|>" 
    )

    return instruction, input, output

def get_test_data(
        sup_image_size, sup_img_bbox, que_image_size, cls_name
):
    instruction = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"You are a highly skilled data annotator, capable of analyzing feature vectors to identify patterns and provide accurate annotations. "
        f"Your current task is to annotate objects within the class '{cls_name}' in an image by identifying their bounding boxes. "
        f"Specifically, you need to output the coordinates of a rectangle that encloses each object of this class. "
        f"The coordinates should be arranged in a clockwise direction and represented as a tuple in the format: (c1, c2), "
        f"where c1 and c2 are the coordinates for the top-left and bottom-right points of the rectangle, respectively. "
        f"The format of the coordinates should be [x_min, y_min, x_max, y_max]. "
        f"Ensure that all coordinate values are within the boundaries of the image size."
        f"\nThe input include the image size and the image in a structured JSON format like so:"
        f"\n"
        f"\nInput:"
        f"\n{{"
        f"\n  \"Image Size\": {sup_image_size},"
        f"\n  \"Image\": <visual_sup>"
        f"\n}}"
        f"\nwhere the position of <visual_sup> represents the image."

        f"\n\nYour output should include the bounding box coordinates and their corresponding class in a structured JSON format like so:"
        f"\n{{"
        f"\n  \"class\": \"{cls_name}\","
        f"\n  \"bounding_box\": {sup_img_bbox}"
        f"\n}}"
        f"<|eot_id|>"
    )

    input = (
        f"<|start_header_id|>user<|end_header_id|>"
        f'{{'
        f'\n  "Image Size":{que_image_size},'
        f'\n  "Image": <visual_que>'
        f'\n}}'
        f'<|eot_id|>'
        f'<|start_header_id|>assistant<|end_header_id|>'
    )

    return instruction, input

def get_train_data_for_llama2(sup_image_size, sup_img_bbox, que_image_size, que_img_bbox, cls_name):
    instruction = (
        f"<s>[INST] <<SYS>>\n"
        f"You are a highly skilled data annotator, capable of analyzing feature vectors to identify patterns and provide accurate annotations. "
        f"Your current task is to annotate objects within the class '{cls_name}' in an image by identifying their bounding boxes. "
        f"Specifically, you need to output the coordinates of a rectangle that encloses each object of this class. "
        f"The coordinates should be arranged in a clockwise direction and represented as a tuple in the format: (c1, c2), "
        f"where c1 and c2 are the coordinates for the top-left and bottom-right points of the rectangle, respectively. "
        f"The format of the coordinates should be [x_min, y_min, x_max, y_max]. "
        f"Ensure that all coordinate values are within the boundaries of the image size."
        f"\nThe input include the image size and the image in a structured JSON format like so:"
        f"\n"
        f"\nInput:"
        f"\n{{"
        f"\n  \"Image Size\": {sup_image_size},"
        f"\n  \"Image\": <visual_sup>"
        f"\n}}"
        f"\nwhere the position of <visual_sup> represents the image."

        f"\n\nYour output should include the bounding box coordinates and their corresponding class in a structured JSON format like so:"
        f"\n{{"
        f"\n  \"class\": \"{cls_name}\","
        f"\n  \"bounding_box\": {sup_img_bbox}"
        f"\n}}"
        f"\n<</SYS>>\n\n"
    )

    input = (
        f'{{'
        f'\n  "Image Size":{que_image_size},'
        f'\n  "Image": <visual_que>'
        f'\n}}'
    )

    output = ( 
        f'\n[/INST]'
        f"{{"
        f"\n  \"class\": \"{cls_name}\","
        f"\n  \"bounding_box\": {que_img_bbox}"
        f"\n}}"
        f"\n</s>"
    )

    return instruction, input, output