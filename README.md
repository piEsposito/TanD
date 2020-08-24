# TanD - Train and Deploy

TanD is a simple, no-code, flexible and customizable framework to automatize the Machine Learning workflow. 

With TanD you can go through the whole ML workflow without writing a single line of code: by creating a project template and setting some configurations on a `.json` file you are able to train a ML model of your choice, store it to `mlflow` to control its lifecycle and create a ready-to-deploy API to serve your it.

Although TanD lets you run your workflows (from train to deploy) with no code at all, it is highly customizable, letting you introduce your chunks of code to enhance your modelling pipelines in anyway you want.

Our mission is to let you avoid repetitive tasks so you can focus on what matters. TanD brings Machine-Learning laziness to a whole new level.

## Rodamap 
The project's roadmap (which is not defined in order of priority) is:
 * Create project templates (`torch` and `sklearn`) for regression tasks in structured data;
 * ~Create a `Dockerfile` in project templates to ease deployment~ OK;
 * ~Create a `cron` job in Docker to update model parameters~ OK;
 * Create tutorials for train and deploy with `tand`;
 * Create project templates (`torch` / `transformers`) for classification tasks in text data;
 * Create project templates (`torch`) for classification in image data;
 * Create `documentation` for the project
 
 # Index
 * [Install](#Install)
 * [Documentation](#Documentation)
 
 ## Install

To install `tand` you can use pip command:

```
pip install train-and-deploy
```

You can also clone the repo and `pip install .` it locally:

```
git clone https://github.com/piEsposito/TanD.git
cd TanD
pip install .
```

 ## Documentation
Documentation for `tand.util` and explanation of project templates:
 * [util](doc/util.md)



---

###### Made by Pi Esposito
