import os
import json
from collections import defaultdict

def load_ids(txt_file):
    """从官方 txt 文件里，按行读取 image_id，存到 set 中。"""
    s = set()
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            image_id = line.strip()
            if image_id:
                s.add(image_id)
    return s

def parse_image_id_from_path(image_path):
    """
    给定完整路径，比如:
      /data2/lihl/data/VOCdevkit/VOC2012/JPEGImages/2012_000005.jpg
    返回它的图像 ID（不含后缀），如 "2012_000005"。
    也可能有的文件名是 "2008_000008.jpg" 等。
    """
    fname = os.path.basename(image_path)  # => 2012_000005.jpg
    img_id = os.path.splitext(fname)[0]   # => 2012_000005
    return img_id


def prepare_voc2012_jsonl(
    input_jsonl,
    trainval_txt,
    test_txt,
    out_trainval_jsonl,
    out_test_jsonl
):
    """
    将 /data/lihl/LLaFS2/data/sft_data_without_attr/all_class_dataset_with_description.jsonl 中
    路径含 "VOC2012" 的记录取出，(image_path, class_name) 级别合并 annotation，并拆分到
    trainval/test 两个 jsonl 文件中。
    """
    # 读取官方划分 ID 列表
    trainval_ids = load_ids(trainval_txt)  # VOC2012 里的 trainval.txt
    test_ids     = load_ids(test_txt)      # VOC2012 里的 test.txt (如果没有，可自行替换成 val.txt)

    # 一个字典，用来按 (image_path, class_name) 分组
    # 格式: merged_dict[(img_path, cls_name)] = {
    #   "image_path":  ...,
    #   "class_name":  ...,
    #   "image_size":  [w, h],
    #   "captions":    [所有描述...],
    #   "annotations": [ [x1, y1, x2, y2], ... ]
    # }
    merged_dict = defaultdict(lambda: {
        "image_path": None,
        "class_name": None,
        "image_size": None,
        "captions": [],
        "annotations": []
    })

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "VOC2012" not in item["image_path"]:
                # 只保留 2012 的
                continue

            img_path = item["image_path"]
            cls_name = item["class"]   # 原json里是 "class"，您也可以改成 "class_name" 
            ann      = item["annotation"]   # [x1, y1, x2, y2]
            img_size = item["image_size"]    # [w, h]
            desc     = item["description"]

            key = (img_path, cls_name)
            if merged_dict[key]["image_path"] is None:
                merged_dict[key]["image_path"] = img_path
                merged_dict[key]["class_name"] = cls_name
                merged_dict[key]["image_size"] = img_size

            merged_dict[key]["captions"].append(desc)
            merged_dict[key]["annotations"].append(ann)

    # 把 (image_path, class_name) 合并后的结果，再根据官方的trainval/test拆分
    voc_trainval = []
    voc_test = []

    for (img_path, cls_name), info in merged_dict.items():
        # 随便取一个 caption
        if len(info["captions"]) > 0:
            caption = info["captions"][0]
        else:
            caption = ""

        # 构造最终输出结构
        result = {
            "image_path":  info["image_path"],
            "class_name":  info["class_name"],
            "annotations": info["annotations"],  # 形状 (N, 4)
            "image_size":  info["image_size"],   # [w, h]
            "caption":     caption
        }

        # 判断属于哪部分
        img_id = parse_image_id_from_path(img_path)
        if img_id in trainval_ids:
            voc_trainval.append(result)
        elif img_id in test_ids:
            voc_test.append(result)
        # 如果既不在trainval也不在test，可忽略

    # 写出
    with open(out_trainval_jsonl, "w", encoding="utf-8") as f_w:
        for r in voc_trainval:
            f_w.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_test_jsonl, "w", encoding="utf-8") as f_w:
        for r in voc_test:
            f_w.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 下面路径请根据自己本地实际情况修改
    input_jsonl = "/root/fsod/data/voc/all_class_dataset_with_description.jsonl"
    trainval_txt = "/root/dataset/voc/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    test_txt     = "/root/dataset/voc/VOCdevkit/VOC2012/ImageSets/Main/val.txt"

    out_trainval_jsonl = "/root/fsod/data/voc/voc2012_trainval.jsonl"
    out_test_jsonl     = "/root/fsod/data/voc/voc2012_test.jsonl"

    prepare_voc2012_jsonl(
        input_jsonl,
        trainval_txt,
        test_txt,
        out_trainval_jsonl,
        out_test_jsonl
    )
    print("Done! Generated:", out_trainval_jsonl, out_test_jsonl)