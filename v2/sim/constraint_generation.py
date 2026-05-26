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
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            ],
        }]

    def _parse_and_save_constraints(self, output, save_dir):
        # Split GPT-4o response into named function blocks, then group by
        # everything-before-the-last-underscore (e.g. stage1_subgoal_constraint{1,2,3}
        # all land in stage1_subgoal_constraints.txt).
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                functions[name] = lines[start:i+1]
        groupings = dict()
        for name in functions:
            key = "_".join(name.split("_")[:-1])
            groupings.setdefault(key, []).append(name)
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")

    def _parse_other_metadata(self, output):
        data_dict = dict()

        def _find(template, key):
            for line in output.split("\n"):
                m = parse.parse(template, line)
                if m is not None:
                    return m[key]
            return None

        num_stages = _find("num_stages = {num_stages}", "num_stages")
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages)

        for k in ("grasp_keypoints", "release_keypoints"):
            raw = _find(f"{k} = {{{k}}}", k)
            if raw is None:
                raise ValueError(f"{k} not found in output")
            data_dict[k] = [int(x.strip()) for x in raw.replace("[", "").replace("]", "").split(",")]

        # we default top_down for every stage.
        raw = _find("approach_directions = {approach_directions}", "approach_directions")
        if raw is None:
            data_dict['approach_directions'] = ["top_down"] * data_dict['num_stages']
        else:
            items = raw.replace("[", "").replace("]", "").split(",")
            data_dict['approach_directions'] = [s.strip().strip('"').strip("'") for s in items]

        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        slug = instruction.split("\n", 1)[0].lower().replace(" ", "_")[:80]
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + slug
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])

        messages = self._build_prompt(image_path, instruction)
        stream = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            stream=True,
        )
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')

        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        self._parse_and_save_constraints(output, self.task_dir)
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir
