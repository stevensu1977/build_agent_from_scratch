import boto3, json, math

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

tool_list = [
    {
        "toolSpec": {
            "name": "cosine",
            "description": "Calculate the cosine of x.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to pass to the function."
                        }
                    },
                    "required": ["x"]
                }
            }
        }
    }
]

message_list = []

initial_message = {
    "role": "user",
    "content": [
        { "text": "What is the cosine of 7?" } 
    ],
}

message_list.append(initial_message)

response = bedrock.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=message_list,
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0
    },
    toolConfig={
        "tools": tool_list
    },
    system=[{"text":"You must only do math by using a tool."}]
)

response_message = response['output']['message']
print(json.dumps(response_message, indent=4))
message_list.append(response_message)
