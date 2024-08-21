import boto3, json,sys

import re
import httpx


session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

models={
    "claude3":"anthropic.claude-3-sonnet-20240229-v1:0",
     "claude3.5":"anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
model_ids=["claude3","claude3.5"]
current_model="claude3"


if len(sys.argv)>1 and sys.argv[1] in model_ids:
    current_model=sys.argv[1]

# define tools_list
tool_list = [
    {
        "toolSpec": {
            "name": "wikipedia",
            "description": "Search Wikipedia for a given query.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "The query string to search on Wikipedia."
                        }
                    },
                    "required": ["q"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate. if is math question use python math packge, etg. math.cos, math.sin "
                        }
                    },
                    "required": ["operation"]
                }
            }
        }
    }
]

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
You have 2 tools calculate and wikipedia, You must only do math by using calcualte tool, others use wikipedia
You must only do math and information search  by using a tool ,
if is math question use full package , etg. math.cos, math.sin 
Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary


wikipedia:
e.g. wikipedia: Django
Returns a summary from searching Wikipedia

Always look things up on Wikipedia if you have the opportunity to do so.

Example session:

Question: What is the cosine of 7?
Thought: I should look up python math page found solution
Action: calculate: math.cos(7)
PAUSE 

Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: France
PAUSE 

You will be called again with this:

Observation: France is a country. The capital is Paris.
Thought: I think I have found the answer
Action: Paris.
You should then call the appropriate action and determine the answer from the result

You then output:

Answer: The capital of France is Paris

Example session

Question: What is the mass of Earth times 2?
Thought: I need to find the mass of Earth on Wikipedia
Action: wikipedia : mass of earth
PAUSE

You will be called again with this: 

Observation: mass of earth is 1,1944×10e25

Thought: I need to multiply this by 2
Action: calculate: 5.972e24 * 2
PAUSE

You will be called again with this: 

Observation: 1,1944×10e25

If you have the answer, output it as the Answer.

Answer: The mass of Earth times 2 is 1,1944×10e25.

Now it's your turn:
""".strip()


class Agent:
    def __init__(self, client, system: str = "",verbose=False) -> None:
        self.client = client
        self.system = system
        self.verbose = verbose
        self.messages: list = []
        print(f'\nModel:{current_model}')
        

    def __call__(self, message=""):
        if message:
            self.messages.append({
                "role": "user",
                "content": [
                    { "text": message } 
                ],
            })
        else:
            return ""
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    
    def execute(self):
        print("\nAgent thinking......")
        if self.verbose:
            print("===============execute start================")
            print(json.dumps(self.messages, indent=4))
            print("===============execute end================")
        
        while True:  # process it until stop reason is not tool_use
            response = self.client.converse(
                #modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                modelId=models[current_model],
                messages=self.messages,
                inferenceConfig={
                    "maxTokens": 2000,
                    "temperature": 0
                },
                toolConfig={
                    "tools": tool_list
                },
                system=[{"text": self.system},{"text":"You must only do math by using a tool."}]
            )
            
            response_content_blocks = response['output']['message']['content']
            print(f'========LLM Result========\n{response_content_blocks[0]["text"]}\n\n')
            
            stop_reason = response['stopReason']
            if self.verbose:
                print(json.dumps(response, indent=4))
                print(f"stopReason: {stop_reason}")

            if stop_reason == 'tool_use':
                
                follow_up_content_blocks = []
                for content_block in response_content_blocks:
                    if 'toolUse' in content_block:
                        tool_use_block = content_block['toolUse']
                        tool_use_name = tool_use_block['name']
                        
                        print(f"Using tool {tool_use_name} {tool_use_block['input']}")
                        tool_result_value = None

                        if tool_use_name == 'calculate':
                            tool_result_value = calculate(tool_use_block['input']['operation'])
                        elif tool_use_name == 'wikipedia':
                            tool_result_value = wikipedia(tool_use_block['input']['q'])
                        
                        if tool_result_value is not None:
                            follow_up_content_blocks.append({
                                "toolResult": {
                                    "toolUseId": tool_use_block['toolUseId'],
                                    "content": [
                                        {
                                            "json": {
                                                "result": tool_result_value
                                            }
                                        }
                                    ]
                                }
                            })

                follow_up_message = {
                    "role": "user",
                    "content": follow_up_content_blocks,
                }
                self.messages.append(response['output']['message'])
                self.messages.append(follow_up_message)

                if self.verbose:
                    print(json.dumps(self.messages, indent=4))
            else:
                break

        return response['output']['message']['content']

def wikipedia(q):
    print('##wikipedia invoked##')
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]
#
def calculate(operation: str) -> float:
    print('##calculate invoked##')
    return eval(operation)




def loop(max_iterations=10, query: str = ""):

    agent = Agent(client=bedrock, system=system_prompt)

    tools = ["calculate", "wikipedia"]

    next_prompt = query

    i = 0
  
    while i < max_iterations:
        i += 1
        result = agent(next_prompt)[0]['text']
        
        print(f"========Agent Step{i}========\n{result}\n\n")

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            if len(action)==0:
                print("Not found action")
                return 
            chosen_tool = action[0][0]
            arg = action[0][1]
            
            if chosen_tool in tools:
                result_tool = eval(f"{chosen_tool}('{arg}')")
                next_prompt = f"Observation: {result_tool}"
                print(f"[Agent] Observation: {result_tool}")

            else:
                next_prompt = "[Agent] Observation: Tool not found"

            # print(next_prompt)
            continue

        if "Answer" in result:
            break

loop(query="What is current age of Mr. Nadrendra Modi multiplied by 2?")
#loop(query="What is the cosine of 7?")



