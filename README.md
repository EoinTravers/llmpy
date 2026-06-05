#  LLMpy

Beyond all the hype, large language models are a powerful tool in the modern data science toolkit.
`llmpy` (lumpy? yeah, let's go with that) is a small utility package that makes using LLMs in data science work a little bit easier.

The workhorse of LLM-powered data science is the ability to take unstructured text data
and turn it into useable data, using simple labels, or fancy structured output schemas.
`llmpy` makes doing this in parallel, over a whole column of values, easy.


```python
# Setup
import os
from dotenv import load_dotenv
from llmpy import OpenAIClient
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm_client = OpenAIClient(api_key=OPENAI_API_KEY, model="gpt-5.4-mini")
```


```python
# Get some data
from re import L
from datasets import load_dataset
import itertools
import pandas as pd

ds = load_dataset("Yelp/yelp_review_full", streaming=True, split="train")
review_df = pd.DataFrame(itertools.islice(ds, 20))
for txt in review_df['text'].iloc[:2]:
    print(txt)
```



    dr. goldberg offers everything i look for in a general practitioner.  he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first.  really, what more do you need?  i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank.
    Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars.


## Example 1: Simple text output


```python
system_prompt = """
Classify the following yelp review as either 'Positive', 'Negative', or 'Neutral'.
Output only the label.
""".strip()

# Classify a single value
one_result = llm_client.call(system_prompt=system_prompt, user_prompt=review_df["text"].iloc[0])
print(one_result)
```

    Positive



```python
# Classify everything in parallel
results = await llm_client.call_many(
    system_prompt=system_prompt, user_prompt=review_df["text"], max_requests_per_minute=100
)
print(results)
```


    ['Positive', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative', 'Neutral', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive']


    


### Example 2: Structured outputs


```python
# Classification restricting output to valid options
from typing import Literal

class Sentiment(BaseModel):
    value: Literal['Positive', 'Negative', 'Neutral']

structured_results = await llm_client.call_many(
    system_prompt=system_prompt, user_prompt=review_df["text"],
    max_requests_per_minute=100,
    response_format=Sentiment
)
print(structured_results)
```

    

    [Sentiment(value='Positive'), Sentiment(value='Negative'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Negative'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Negative'), Sentiment(value='Neutral'), Sentiment(value='Neutral'), Sentiment(value='Negative'), Sentiment(value='Negative'), Sentiment(value='Positive'), Sentiment(value='Negative'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Positive'), Sentiment(value='Positive')]


    



```python
results = [r.value for r in structured_results]
print(results)
```

    ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Neutral', 'Neutral', 'Negative', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive']



```python
# Richer structure
from pydantic import Field
class ReviewAnnotations(BaseModel):
    subject_name: str = Field(description="Name of the person or place being reviewed")
    subject_type: str | None = Field(description="What kind of person/place is being reviewed? Leave blank if unknown")
    sentiment: Literal['Positive', 'Negative', 'Neutral']

structured_results = await llm_client.call_many(
    system_prompt="Extract the information required", user_prompt=review_df["text"],
    max_requests_per_minute=100,
    response_format=ReviewAnnotations
)
result_df = pd.DataFrame([r.model_dump() for r in structured_results])
result_df['text'] = review_df['text']
result_df

```

    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_name</th>
      <th>subject_type</th>
      <th>sentiment</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dr. goldberg</td>
      <td>general practitioner</td>
      <td>Positive</td>
      <td>dr. goldberg offers everything i look for in a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dr. Goldberg</td>
      <td>doctor</td>
      <td>Negative</td>
      <td>Unfortunately, the frustration of being Dr. Go...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dr. Goldberg</td>
      <td>doctor</td>
      <td>Positive</td>
      <td>Been going to Dr. Goldberg for over 10 years. ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dr. Goldberg</td>
      <td>doctor</td>
      <td>Positive</td>
      <td>Got a letter in the mail last week that said D...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dr. Goldberg</td>
      <td>doctor/office</td>
      <td>Negative</td>
      <td>I don't know what Dr. Goldberg was like before...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>doctor</td>
      <td>doctor</td>
      <td>Positive</td>
      <td>Top notch doctor in a top notch practice. Can'...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Dr. Eric Goldberg</td>
      <td>doctor</td>
      <td>Positive</td>
      <td>Dr. Eric Goldberg is a fantastic doctor who ha...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dr. Goldberg</td>
      <td>Doctor</td>
      <td>Negative</td>
      <td>I'm writing this review to give you a heads up...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Wing sauce</td>
      <td>food</td>
      <td>Negative</td>
      <td>Wing sauce is like water. Pretty much a lot of...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>golf range</td>
      <td>place</td>
      <td>Neutral</td>
      <td>Decent range somewhat close to the city.&nbsp;&nbsp;The ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>this place</td>
      <td>driving range</td>
      <td>Negative</td>
      <td>Owning a driving range inside the city limits ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>This place</td>
      <td>NaN</td>
      <td>Negative</td>
      <td>This place is absolute garbage...&nbsp;&nbsp;Half of the...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>the range</td>
      <td>place</td>
      <td>Positive</td>
      <td>I drove by yesterday to get a sneak peak.&nbsp;&nbsp;It ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>this store</td>
      <td>store</td>
      <td>Negative</td>
      <td>After waiting for almost 30 minutes to trade i...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>This place</td>
      <td>place</td>
      <td>Positive</td>
      <td>This place was DELICIOUS!!&nbsp;&nbsp;My parents saw a r...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Fish Sandwich</td>
      <td>food item</td>
      <td>Positive</td>
      <td>Can't miss stop for the best Fish Sandwich in ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>This place</td>
      <td>restaurant</td>
      <td>Positive</td>
      <td>This place should have a lot more reviews - bu...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Old school</td>
      <td>restaurant</td>
      <td>Positive</td>
      <td>Old school.....traditional \"mom 'n pop\" qual...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>fish sandwich</td>
      <td>food item</td>
      <td>Positive</td>
      <td>Good fish sandwich.</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Emil's</td>
      <td>restaurant</td>
      <td>Positive</td>
      <td>After a morning of Thrift Store hunting, a fri...</td>
    </tr>
  </tbody>
</table>



# Development

Run tests:
```bash
uv run pytest
```

Run linter:
```bash
uv run ruff check src tests
```

Run type checker:
```bash
uv run mypy src
```

