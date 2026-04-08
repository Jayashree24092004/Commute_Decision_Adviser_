---
title: Commute Decision Advisor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Commute Decision Advisor. OpenEnv Environment

This is a place where computers can learn to make decisions about how to get to work or school. They have to think about things like traffic, tolls and what time they need to be

It was made for a contest with Meta, PyTorch and Scaler.

---

## Why This Is Important

Planning your commute is not easy. You have to think about:

* Getting to school. Work on time

* Avoiding traffic and tolls

* Making sure you can do everything you need to do

* Going to places in one trip like dropping off your kids and then going to the store

This environment is like a simulator that helps computers learn how to make good decisions in these situations.

---

## What The Environment Looks Like

The environment is like a city called Synthcity. It has:

* Places where people live, like homes and suburbs

* Places where people work, like offices and universities

* Stores and services like hospitals and pharmacies

* Train stations and airports

* kinds of roads like highways and local streets

Computers have to figure out the best way to get around this city.

---

## How To Use The Environment

This environment follows the OpenEnv rules:

* You can start a trip with a POST request to /reset

* You can take an action. Get a reward with a POST request to /step

* You can get the state with a GET request to /state

* You can get a list of tasks with a GET request to /tasks

* You can check the system status with a GET request to /health

* You can use the UI with a GET request to /web

---

## What The Computer Needs To Do

The computer needs to send a message that says:

```

{

"chosen_route": "Route A or Route B or Route C"

"reasoning": "why they chose that route"

"stop_order": ["School" "Pharmacy"]

"arrival_times": {"School": "8:22"}

}

```

Some things to note:

* Easy tasks just need the chosen route

* Hard tasks need the stop order and arrival times

* The computer always needs to say why it made its decision

---

## What The Environment Sends Back

The environment sends back a message that says:

```

{

"task_id": "M001"

"difficulty": "

"scenario": "..."

"start": "Home"

"end": "Office"

"depart_time": "8:15"

"time_slot": "morning"

"route_options": [...]

"constraints": [...]

"stops": []

"city_context": {...}

"task_prompt": "what the computer needs to do"

}

```

---

## Tasks And Difficulty Levels

Easy tasks are like this: the computer just needs to choose the route. There are three tasks.

Medium tasks are like this: the computer needs to choose the route but there are some rules it has to follow. There are three tasks.

Hard tasks are like this: the computer needs to plan a trip with multiple stops. There are three tasks.

---

## How The Computer Gets Rewarded

The computer gets a reward between 0 and 1.

Easy tasks use this formula: 0.6 times the route, plus 0.4 times the quality of the reasoning.

Medium tasks use this formula: 0.7 times the rules followed plus 0.3 times the quality of the reasoning.

Hard tasks use this formula: 0.5 times the rules followed. 0.2 Times the number of stops plus 0.3 times the quality of the reasoning.

Some key things about the reward system:

* It checks the rules in a way

* The computer gets credit for trying

* It gets penalized for breaking the rules

---

## What Makes This Environment Special

* It uses a graph to simulate the city

* It always gives the computer three route options

* It models traffic in a way

* It takes into account things like store hours and road closures

* It helps the computer plan the route with multiple stops

---

## How To Set Up The Environment

Step 1: Clone the repository

git clone <your-repo-url>

cd commute-advisor

Step 2: Install the dependencies

pip install -r requirements.txt

Step 3: Run the environment

main:app --host 0.0.0.0 --port 7860

Step 4: Open the UI

http://localhost:7860/web

---

## How To Use Docker

Build the image

docker build -t commute-env.

Run the container

docker run -p 7860:7860 commute-env

---

## How To Run The Baseline Agent

Run the agent using

python inference.py

It will output something like this:

[START] task=E001 env=commute model=mistral

[STEP] step=1 action=Route A reward=0.80 done=true error=null

[END] success=true steps=1 score=0.80 rewards=0.80

---

## Baseline Scores

Easy: 0.85

Medium: 0.72

Hard: 0.61

The overall score is 0.73

---

## Example Scenario

Imagine you need to drop off your kid at school avoid toll roads and get to work on time. The computer needs to choose the route follow all the rules and explain why it made its decision.

---

## Project Structure

main.py is the main file

models.py defines the types of actions and observations

city_graph.py defines the city

route_engine.py handles route generation

grader.py implements the reward system

tasks.py defines all the tasks

inference.py runs the baseline agent

openenv.yaml contains environment metadata

Dockerfile enables containerization

web_ui.html provides an interface

---

## Hugging Face Deployment

The environment is deployed as a Docker-based Hugging Face Space and supports both API interaction and browser-based UI.

---

## What Makes This Project Innovative

* It combines types of rewards

* It guarantees that the computer will get route options

* It models traffic and store hours in a way

* It helps the computer plan the route with multiple stops

---

## Future Improvements

* Integrate real-time traffic data

* Use machine learning to adapt the routes

* Make the city graph bigger and more complex

* Add scenarios with computers working together

---

##

This environment is a good way to test computers on real-world planning problems. It combines reinforcement learning, structured reasoning and real-world constraints, into one system.

---


<img width="945" height="626" alt="image" src="https://github.com/user-attachments/assets/18b3f992-efd4-49d5-b40b-459e433e212b" />
