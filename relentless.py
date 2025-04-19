import asyncio
from graph import run_, ron_, get_task_stat
from db import engine, Graph, Task, Run, get_graph_from_a_folder, get, go, get_by_id
from sqlmodel import Session
import itertools
import random
from tqdm import tqdm

async def callopenai(x: str, model='gemini-2.0-flash',tools: list[Literal['crop']]=[]):
    print(x)
    parts = re.split(ugly, x)
    image_set = []
    content = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
        else:
            buffer = io.BytesIO()
            img = get_image_by_short_hash(part)
            if part in image_set:
                content.append({"type": "text", "text": f"<image_{part}>"})
                continue
            image_set.append(part)
            print(part)

            max_dim = 2000
            if max(img.width, img.height) > max_dim:
                scale = max_dim / max(img.width, img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"}
            })
            content.append({"type": "text", "text": f"<image_{part}>"})
    
    def retrieve(image_id):
        idc = re.match(r"image_(\d+)", image_id)
        if idc:
            image_id = int(idc.group(1)) - 1
            return get_image_by_short_hash(image_set[image_id])
        hs = re.match(ugly, image_id)
        if hs:
            return get_image_by_short_hash(hs.group(1))
        try:
            image_id = int(image_id)
            return image_set[image_id]
        except:
            pass
        return image_set[0]
    
    def crop(image_id=0, x1=0, y1=0, x2=1000, y2=1000):
        image = retrieve(image_id)
        image_dims = image.size
        x1 = x1 / 1000 * image_dims[0]
        y1 = y1 / 1000 * image_dims[1]
        x2 = x2 / 1000 * image_dims[0]
        y2 = y2 / 1000 * image_dims[1]
        return image.crop((x1, y1, x2, y2))
    
    tools = []
    if image_set and 'crop' in tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "crop",
                    "description": "Crop an image",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string", "description": f"use 'image_1' to refer to the 1st image, or its representation, '<image_{image_set[0]}>'"},
                            "x1": {"type": "number", "description": "coordinates are from 0 to 1000"},
                            "y1": {"type": "number"},
                            "x2": {"type": "number"},
                            "y2": {"type": "number", "description": "coordinates are from 0 to 1000"},
                        },
                        "required": ["x1", "y1", "x2", "y2"]
                    }
                }
            }
        )

    messages=[{
        'role': 'user',
        'content': content
    }]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    while response.choices[0].message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": [tool_call]
        })
        tool_call = response.choices[0].message.tool_calls[0]
        func_name = tool_call.function.name
        if func_name == "crop":
            args = json.loads(tool_call.function.arguments)
            result = crop(**args)
            repr = img_go(result)
            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            messages.append({
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": repr
                    }
                ],
                "tool_call_id": tool_call.id
            })
        else:
            raise ValueError(f"Unknown tool: {func_name}")

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    
    return response.choices[0].message.content


async def bombarda(run):
    with Session(engine) as session:
        session.add(run)
        graph_orig = run.graph
        task = run.task
        runs = [run]
        graph_ids = []
        relentless = get_graph_from_a_folder('sampo/bflow', groph=True)
        for _ in range(100):
            print('LEN: ', len(runs))
            ron = await ron_(relentless, [runs[0]])
            graph_ids.append(ron.new_graph_id)
            graph_ = get_by_id(Graph, ron.new_graph_id)
            run = await run_(graph_, task)
            runs.append(run)
            if run.correct:
                break

def test_bob():
    ts = get_task_stat()
    ts = {k for k, v in ts.items() if v[0] == 1 and v[1] > 2 }

    cot_graph = get_graph_from_a_folder('sample/basic')
    tss = get(Run, Graph)[cot_graph.id]

    for t in tss:
        if t.id in ts:
            print('FOUND!')
            asyncio.run(bombarda(cot_graph, t.task))
            exit
    
    asyncio.run(bombarda(asyncio.run(run_(cot_graph, get_by_id(Task, list(ts)[0])))))
    

    print(len(ts))

async def merge(graph1, graph2):
    runs = get(Run, Graph)
    run1 = runs[graph1.id]
    run2 = runs[graph2.id]
    tasks = list(set([x.task_id for x in itertools.chain(run1, run2) if not x.correct]))
    merg = get_graph_from_a_folder('sampo/merger', groph=True)
    go(merg)
    for _ in tqdm(range(100)):
        ron = await ron_(merg, [run1[0], run2[0]])
        new_graph = ron.new_graph
        go(new_graph)
        tasks = random.sample(tasks, 3)
        sum_correct = 0
        for task_id in tasks:
            current_task = task(task_id)
            run = await run_(new_graph, current_task)
            go(run)
            sum_correct += run.correct
        if sum_correct == 3:
            break

if __name__ == '__main__':
    test_bob()
    # merge(get_graph_from_a_folder('sample/basic'), get_graph_from_a_folder('/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7'))