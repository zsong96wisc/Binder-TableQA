import openai, time

client = openai.OpenAI()


def call_llm(message: str) -> str:
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message},
        ],
    )

    if result.choices[0].finish_reason.strip() == "stop":
        return result.choices[0].message.content
    else:
        print(result)
        time.sleep(30)
        raise Exception("Error Occurs")
