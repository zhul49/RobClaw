import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime


def encode_image(image_path):
    # read a PNG from disk and return its bytes as a base64 string since OpenAI chat-completions API accepts images as data-URLs
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ConstraintGenerator:
    def __init__(self, config):
        # One-time setup. Reads the OpenAI API key from the environment
        # and loads the prompt template from disk.
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        # a long string loaded once at construction that contains instructions filled in and template
        prompt_text = self.prompt_template.format(instruction=instruction)
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        # message structure we send into OpenAI
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")

    def _parse_other_metadata(self, output):
        # add 3 more metadata: num_stages, grasp_keypoints ([3, -1, -1] ), release_keypoints ([-1, -1, 3]) from GPT's reply
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints. e.g. '[0, -1, -1]' → [0, -1, -1]
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints (same shape as grasp_keypoints)
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        # meta data contains init_keypoint_positions (list of [x,y,z]), num_keypoints, keypoint_shapes ({"0": "can carton jar", ...})
        slug = instruction.split("\n", 1)[0].lower().replace(" ", "_")[:80]
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + slug
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build the chat-completions message list
        messages = self._build_prompt(image_path, instruction)
        # stream the response so we can show progress
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=self.config['temperature'],
                                                        max_tokens=self.config['max_tokens'],
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save the raw response
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # Pprse out the function bodies and stage metadata
        self._parse_and_save_constraints(output, self.task_dir)
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir
