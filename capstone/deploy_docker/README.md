## Using a LLM with Ollama Api for Triviaqa

In this project, the [llama3.1 with 8B parameters](https://huggingface.co/meta-llama/Llama-3.1-8B) 
large language models (LLM) is used to answer questions from the
 [triviaqa](https://github.com/mandarjoshi90/triviaqa) dataset. 
 [Ollama](https://ollama.com/), which is an open-source tool that lets users run 
 LLMs locally on their computer, is used to download the LLM locally and use it 
 for qa. One of the files from the dataset which contains both the trivia questions 
 and answers will be used for testing. The answers obtained from the LLM, will then
  be compared to expected answers in the triviaqa dataset to evaluate performance 
  using a [BLEU](https://aclanthology.org/P02-1040.pdf)-1gram (Bilingual Evaluation 
  Understudy) score. 

The following files are used to create a docker image using a Windows11 WSL2 shell and the Docker Desktop. File list for docker image as in the git repository are as below:
* triviaqa_ijson_utils.py:  ijson utilities to get the qa entries from triviaqa json files into a list of dict items
* app_ijson_ollama.py: main program to perform qa using a open source llm model
* static folder with style.css file for flask app
* templates folder with index.html template
* requirements.txt: holds packages and version information for creating a docker image
* Dockerfile: has steps to create a docker image
* .env : has environment variables to run the image with different settings


To create a docker build</br>
docker build -t \<your-dockerhub-username\>/\<imagename\>:\<tag\> .

The image may be pushed to the [Docker Hub](https://hub.docker.com/) to run in a Docker Container
