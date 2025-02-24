def get_prompt_v2(class_name, image_size, annotations, caption):
    """
    根据输入数据动态生成 Prompt。

    Args:
        class_name (str): 目标类别名称。
        image_size (list): 图像大小，格式为 [width, height, channels, depth]。
        annotations (list): 物体标注信息，格式为 [[x_min, y_min, x_max, y_max], ...]。
        caption (str): 图像描述。

    Returns:
        tuple: 包含 instruction, input 和 output，其中 output 为 bbox 的真实值 list。
    """
    instruction = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"You are a highly skilled data annotator, specializing in analyzing JSON-formatted data to extract relevant object annotations accurately. "
        f"Your primary responsibility is to identify and annotate objects in a given image based on the provided data. "
        f"Each image may contain multiple objects belonging to different categories, but your goal is to focus on objects of the class '{class_name}'. "
        f"Annotations are expected to include the class name and bounding box coordinates for each object. "
        f"The bounding box coordinates should be represented in the format [x_min, y_min, x_max, y_max] and must strictly adhere to the image size boundaries. "
        f"\n\nYour task is structured as follows:"
        f"\n1. Analyze the input data to understand the image's size, description, and existing annotations."
        f"\n2. Identify all objects of the class '{class_name}' and extract their bounding box coordinates."
        f"\n3. Ensure that all detected objects of the specified class are annotated correctly, even when there are multiple instances of the same class in the image."
        
        f"\n\nThe input includes the image size, description, and annotations, provided in JSON format as follows:"
        f"\n"
        f"\nInput:"
        f"\n{{"
        f"\n  \"image\": {{an image user give}},"
        f"\n  \"image_size\": {{[img_x_max, img_y_max]}},"
        f"\n  \"description\": \"{{A sentence describing this image.}}\","
        f"\n}}"
        f"\n<|eot_id|>"

        f"\n\nYour output should only include the bounding box coordinates for the class '{class_name}', formatted as follows:"
        f"\nOutput:"
        f"\n[[x_min, y_min, x_max, y_max], ...]"
        f"\n<|eot_id|>"

        f"This image contains some reference information, which includes attributes of certain classes that you may refer to when identifying these categories, along with their appearance in the image. "
        f"These references are one-to-one correspondences for your reference. "
        f"\n\nThe reference information you may use is as follows:"
        f"\n"
        f"\n{{"
        f"\n  \"{class_name}'s\": <attr_embeds_start><attr_embeds_end>,"
        f"\n  \"visual tokens of {class_name}'s attribute\": <visual_sup_start><visual_sup_end>"
        f"\n}}"

    )

    input_text = (
        f"<|start_header_id|>user<|end_header_id|>"
        f"{{"
        f"\n  \"image\": <visual_que_start><visual_que_end>"
        f"\n  \"image_size\": {image_size},"
        f"\n  \"description\": \"{caption}\""
        f"\n}}"
        f"\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

    output = str([list(bbox) for bbox in annotations])
    
    return instruction, input_text, output

def get_prompt_new(sup_data, que_data, cls_name):
    """
    根据输入数据动态生成 Prompt，包括支持数据和查询数据。

    Args:
        sup_data (dict): 支持数据，包含 image_path, image_size, description, annotations 等字段。
        que_data (dict): 查询数据，包含 image_path, image_size, description, annotations 等字段。
        cls_name (str): 目标类别名称。

    Returns:
        tuple: 包含 instruction, input 和 output 的字符串。
    """
    sup_image_size = sup_data["image_size"]
    sup_description = sup_data["description"]
    sup_annotations = [anno for anno in sup_data["annotations"] if anno["category_name"] == cls_name]

    que_image_size = que_data["image_size"]
    que_description = que_data["description"]
    que_annotations = [anno for anno in que_data["annotations"] if anno["category_name"] == cls_name]

    instruction = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"You are a highly skilled data annotator, specializing in analyzing JSON-formatted data to extract relevant object annotations accurately. "
        f"Your primary responsibility is to identify and annotate objects in a given image based on the provided data. "
        f"Each image may contain multiple objects belonging to different categories, but your goal is to focus on objects of the class '{cls_name}'. "
        f"Annotations are expected to include the class name and bounding box coordinates for each object. "
        f"The bounding box coordinates should be represented in the format [x_min, y_min, x_max, y_max] and must strictly adhere to the image size boundaries. "
        f"\n\nYour task is structured as follows:"
        f"\n1. Analyze the input data to understand the image's size, description, and existing annotations."
        f"\n2. Identify all objects of the class '{cls_name}' and extract their bounding box coordinates."
        f"\n3. Ensure that all detected objects of the specified class are annotated correctly, even when there are multiple instances of the same class in the image."
        
        f"\n\nThe input includes the image path, image size, description, and annotations, provided in JSON format as follows:"
        f"\n"
        f"\nInput:"
        f"\n{{"
        f"\n  \"image\": <visual_sup>,"
        f"\n  \"image_size\": {sup_image_size},"
        f"\n  \"description\": \"{sup_description}\","
        f"\n}}"

        f"\n\nYour output should only include the bounding box coordinates and their corresponding classes for the class '{cls_name}', formatted as follows:"
        f"\nOutput:"
        f"\n{{"
        f"\n  \"annotations\": ["
        + "".join(f"\n    {{ \"class\": \"{anno['category_name']}\", \"bbox\": {anno['bbox']} }}," for anno in sup_annotations)
        + f"\n  ]"
        f"\n}}"
        f"\n<|eot_id|>"
    )

    input = (
        f"<|start_header_id|>user<|end_header_id|>"
        f"{{"
        f"\n  \"image\": <visual_que>,"
        f"\n  \"image_size\": {que_image_size},"
        f"\n  \"description\": \"{que_description}\","
        f"\n}}"
        f"\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

    output = (
        f"{{"
        f"\n  \"annotations\": ["
        + "".join(f"\n    {{ \"class\": \"{anno['category_name']}\", \"bbox\": {anno['bbox']} }}," for anno in que_annotations)
        + f"\n  ]"
        f"\n}}"
        f"<|eot_id|>"
    )
    return instruction, input, output

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

if __name__ == "__main__":
    import json
    with open("/data/lihl/LLaFS2/data/sft_data_new_pe/sft_voc_dataset.json", "r") as f:
        data = json.load(f)
    
    print(data[4])
    for i in range(3):
        print('-'*50, f'i = {i}', '-'*50)
        record = data[i]
        instruction, input, output = get_prompt_new(record, data[4],'chair')
        print(f"Image Path: {record['image_path']}")
        print(f"Image Size: {record['image_size']}")
        print(f"Description: {record['description']}")
        print(f"Annotations: {record['annotations']}\n")
        print(f'Instruction: {instruction}')
        print(f'Input: {input}')
        print(f'output: {output}')