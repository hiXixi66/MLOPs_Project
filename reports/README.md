# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [Y] Create a git repository (M5)
* [Y] Make sure that all team members have write access to the GitHub repository (M5)
* [Y] Create a dedicated environment for you project to keep track of your packages (M2)
* [Y] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [Y] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [Y] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [Y] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [Y] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [Y] Do a bit of code typing and remember to document essential parts of your code (M7)
* [Y] Setup version control for your data or part of your data (M8)
* [Y] Add command line interfaces and project commands to your code where it makes sense (M9)
* [Y] Construct one or multiple docker files for your code (M10)
* [Y] Build the docker files locally and make sure they work as intended (M10)
* [Y] Write one or multiple configurations files for your experiments (M11)
* [Y] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [Y] Use profiling to optimize your code (M12)
* [Y] Use logging to log important events in your code (M14)
* [Y] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [Y] Write unit tests related to the data part of your code (M16)
* [Y] Write unit tests related to model construction and or model training (M16)
* [Y] Calculate the code coverage (M16)
* [Y] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [Y] Add a linting step to your continuous integration (M17)
* [Y] Add pre-commit hooks to your version control setup (M18)
* [Y] Add a continues workflow that triggers when data changes (M19)
* [Y] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [Y] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [Y] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [Y] Create a FastAPI application that can do inference using your model (M22)
* [Y] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [Y] Write API tests for your application and setup continues integration for these (M24)
* [Y] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [Y] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

18

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s203957, s232253, s232414, s232472, s243075

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- question 3 fill here ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

In order to get the exact same work environment a new team meamber would have to initialize our github repository and then write 'pip install -e .' (I assume they will want developer rights). This will then install all the libraries created for this project (folders with __init__.py files) and install the libraries with the same version as us (assuming we filled in requirements.txt+requirements_dev.txt thouroughly)

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

From the cookiecutter template we filled out models/ with model parameter weights, configs with the configuration file used to start the training code, src/ with ML code, tests/ with tests for our source code. We created a new folder outputs/ with the configuration files generated from hydra each training run. While the template typically places outputs in directories based on their creation date, we did not utilize this feature to minimize code restructuring. Although this approach worked well for this project, adopting the default structure would be beneficial for longer-term projects in order to minimize chances of overwriting run code.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

To ensure that our project contained readable and reproducible code we used pep8 formatting so that our code followed a certain style which makes it a lot easier to read and to modify since we do not have to adjust to any other formatting styles...

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer: In total, we have implemented 103 tests. Primarily, we are testing data processing functions (e.g., ensuring correct handling of various data input scenarios) and model-related functions (e.g., verifying the model's training and inference behavior), as these are the most critical parts of our application.




--- question 7 fill here ---

> Answer:

In total, we have implemented 103 tests. Primarily, we are testing data processing functions (e.g., ensuring correct handling of various data input scenarios) and model-related functions (e.g., verifying the model's training and inference behavior), as these are the most critical parts of our application.

<pre>
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/rice_images/__init__.py       0      0   100%
src/rice_images/data.py          40     26    35%   13-37, 61-92, 97
src/rice_images/model.py         16      7    56%   33-44
tests/__init__.py                 0      0   100%
tests/test_api.py                 0      0   100%
tests/test_data.py               53      2    96%   97-98
tests/test_model.py              50      0   100%
-----------------------------------------------------------
TOTAL                           159     35    78%
</pre>
### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our project is 78%, which includes all our source code and tests. While this represents a strong level of coverage, we are still far from achieving 100%. Even if we were to achieve 100% code coverage, it would not guarantee that the code is error-free.
Covered Statements = 159 - 35 = 124
Coverage (%)=( 159 / 124 )×100≈77.99% ( 78%)

Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/rice_images/__init__.py       0      0   100%
src/rice_images/data.py          40     26    35%   13-37, 61-92, 97
src/rice_images/model.py         16      7    56%   33-44
tests/__init__.py                 0      0   100%
tests/test_api.py                 0      0   100%
tests/test_data.py               53      2    96%   97-98
tests/test_model.py              50      0   100%
-----------------------------------------------------------
TOTAL                           159     35    78%

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used branches during the production of our project. The structure of our branches was the following: there were mainly two types of branches that more or less had the same functionality. We had personal branches, so a direct copy of the main branch that each one of us could work on and test the performance of our assigned tasks, and the second type which basically defined the task to be done (for example, pip8-complicance, data_storage, etc.). Furthermore, since each one of us practically worked on different parts of the project we did not need to utilize pull requests too often, only in cases where one of the group members modified something without the explicit knowledge of the other group members. Once we got the specific piece of code working on one of the branches we would perform a merge with the main branch and update the code stored there.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did use dvc to manage our data. While it wasn't necessary for this project as data formatting wasn't explored much and the original source for our data remained active, it does in general make sense to back up your data to a server and use dvc such that if the data was overwritten, the original data could always be recovered.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We use continuous integration (CI) to automatically check our code for quality and correctness. We use GitHub Actions and have two main CI workflows: one for linting and one for unit tests. This keeps things organized and makes it easier to see what's happening.

The linting workflow checks our code style using flake8. It makes sure our code looks consistent and follows coding standards. This runs every time someone pushes code or makes a pull request to the main branch. We test on different operating systems (Ubuntu, Windows, and macOS) and a specific Python version (3.12) to make sure our code works everywhere. We also use caching to speed things up by saving downloaded packages. Even if there are linting errors, the CI process keeps running so we can see other problems too.

The unit testing workflow checks if our code actually works correctly using automated tests. Like the linting workflow, it runs on every code push or pull request to the main branch and tests on different operating systems and Python versions. It also uses caching to be faster. This workflow installs everything our code needs and then runs the tests using pytest. We also create coverage reports to see how much of our code is tested.

Having separate workflows for linting and unit testing makes our CI process clearer and more efficient. This helps us write better code that works reliably on different systems.
The Linting workflow can be found : [Linting](https://github.com/hiXixi66/MLOPs_Project/actions/workflows/linting.yaml)


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra to manage configurations such as hyperparmeters. The code would be run by using python src/rice_images/train.py and the configuration for the run would be saved in outputs/<date>/... The hyperparameters could be altered by changing the values in configs/train.yaml.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of config files. Whenever an experiment was run a config file would be generated. In order to reproduce the experiment one could reuse the config file from that experiment by running the code with this config file. Feks python src/rice_images/train.py --config-path ../../outputs/2025-01-17/08-41-56/.hydra --config-name config.yaml


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
>Answer:

We use Docker and the Docker Engine to containerize various parts of our project, including API, backend, evaluation, frontend, and training components. Each component has a Dockerfile defining its environment. These Dockerfiles specify a base image, install necessary dependencies, copy project files, and define the startup command.

To build our Docker images, we use the docker build command. Specifically: `docker build -t backend . -f backend.dockerfile .`. This builds the backend image.

To run the built images, we use the docker run command.
Specifically for the backend: `docker run --rm -p 8000:8000 -e "PORT=8000" backend`. This command maps port 8000 on the host to port 8000 in the container and sets the PORT environment variable inside the container to 8000. The --rm flag removes the container after it stops.

These commands ensure consistent and isolated environments for each component of our project, simplifying development, testing, and deployment.

This is the link for the dockerfiles: [Dockerfiles](https://github.com/hiXixi66/MLOPs_Project/tree/main/dockerfiles)



### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer: 

Debugging methods varied based on the issue encountered during the experiments. For instance, when facing unexpected errors or suboptimal performance, I relied on tools such as logging, assertion checks, and interactive debugging (e.g., using `pdb`). For computational bottlenecks or performance optimization, I conducted profiling to analyze resource usage and identify inefficiencies. 

We used PyTorch’s `torch.profiler` to profile the model during execution. This involved capturing CPU activities, recording tensor shapes, and analyzing call stacks to pinpoint areas of high resource consumption. Additionally, I leveraged TensorBoard visualization through the `tensorboard_trace_handler` to gain insights into model behavior and optimize specific parts of the code. For example, the profiling revealed whether the model was spending excessive time in certain layers or operations, allowing for targeted optimizations.

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used engine, bucket and cloud function. Bucket was used for data storage with version control. Engine was used for training the model (just to try it. It wasn't part of our procedure). Engine and functions were both used to deploy the function. We ended up preffering engine as it was easier...

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We didn't use the Compute Engine with VMs very much. We figured out how to upload and run the code on the VMs but took them down again as soon as our scripts were done, to free up computer resources and save money. When we did use them, hardware was picked close to us (Berlin) and simple CPU architecture was utilized as our code was not trained for performance. The Cloud Run was used to deploy our model and thus we spent most of our time with this service.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Data bucket](figures/q19_bucket.png).

[Frontend-backend bucket](figures/frontend_backend_bucket.png).

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Frontend-backend registry](figures/artifact_registryFB.png).



### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer: [Frontrnd-Backend build](figures/build_fb.png).



### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to complete a training run in the Compute Engine. We did this by connecting to the VM terminal through SSH. Then we installed Python3.12 (requirement from our TOML). Then we cloned the directory into the VM. Lastly the libraries and dependencies were installed as 'pip install -e .'. From there the training python file could be run as normal via CLI inputs like 'python src/rice_images/train.py --lr 1e-4'

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We used FastAPI to write the API for our model. Our api.py file contained the initialization of the application as well as setting up the security of our API using CORS middleware. This allowed us to define which requests we allow to be made to our model (the HTTP methods) as well as which origins are permitted to make cross-origin requests. Furthermore, this file contained a basic health check just to confirm that the API is running properly and it includes all of the individual requests defined in our backend.py file. This file purely defines the functionality of the individual allowed requests. For instance, the GET request defined in backend.py takes an image as an input and puts it through a classification function that returns the probabilities of the specific image being a certain class.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

To deploy the model to the cloud we used the api.py and backend.py files described above. We created a virtual machine instance which we connected to our git repository and installed our entire model and the relevant dependencies. We used the SSH virtual machine terminal to do this and to start up the API itself using the following command: uvicorn api:app --host 0.0.0.0 --port 5000. Once the API was running smoothly we could invoke requests to our model using a local terminal (for instance GitBash). We could first check if the connection to the virtual machine works properly by checking the status of our application by running the following command: curl http://34.123.45.67:5000/health (of course replacing the external IP with the current relevant one). Furthermore, if we wanted to test our model and see how it performs on a random image of rice we could invoke the GET request using the following command: curl -X POST http://34.79.157.188:5000/classify/ \
-H "Content-Type: multipart/form-data" \
-F "file=@path/c/MASTER/ML Operations/MLOPs_Project/data/raw/Rice_Image_Dataset/Rice_Image_Dataset/Arborio/Arborio (1).jpg"" (changing the path and external IP accordingly).



### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We created a test_api.py file which included two tests where we utilized pytest to initialize the test client. Our first test named test_model_with_preprocessed_image performs a test to ensure that the model can correctly provide reasonable class predictions given a random preprocessed image. Before the definition of the test we also define the transformations used for normalization of the data. The second test checks the behaviour of the model in case it is provided with no image file and that it can correctly output the correct error code in such a case.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here --- -andu

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here --- -everyone write what you spend from Billing 
Viktor: 7eur on Compute Engine; 6 eur on Cloud Storage.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
>Answer:
In addition to the backend API, we also created a frontend application to provide a user-friendly interface for interacting with the API. This frontend is deployed on Google Cloud using a Dockerfile (similar to how we deployed the backend). You can access the frontend here: https://frontend-1083855416866.europe-west10.run.app/

The purpose of the frontend is to allow users to easily upload images of rice. Once uploaded, the frontend sends the image to the backend API for classification. The API then analyzes the image and returns the predicted classification along with the probability distribution of all possible classes. The frontend presents this information to the user in a clear and understandable way, typically including:

The predicted class label
A visual representation of the image
A plot illustrating the probability distribution for each class


### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---
[Overview](figures/overview.png)


The starting point of the diagram is our local setup, where we integrate a Python 3.12 development environment, the "Weights & Biases" tracking system, and all dependencies required for ResNet-18 model training into our code. This setup allows us to run experiments locally while seamlessly recording metrics. Additionally, extensive preparation work has been completed to support porting the system to the cloud, including creating training and evaluation models, designing Dockerfiles, profiling, logging, and other essential tasks. Ultimately, we chose ResNet-18, achieving a classification accuracy exceeding 90%.

When code is pre-committed, committed, and pushed to GitHub, it automatically triggers the GitHub Actions workflow. This pipeline is configured to build the project and push the resulting container image to the container registry. The container image contains the latest version of the source code along with its dependencies. From there, the updated container image is pulled by the Google Cloud Platform (GCP), where the ResNet-18 model is executed. After deployment, we developed a human-computer interaction interface, enabling users to input images via a URL link. The model runs on GCP and processes the input, delivering the results back to the user.

The integration with "Weights & Biases" ensures seamless tracking and monitoring of experimental metrics and outcomes, providing valuable insights during the development and optimization phases. Additionally, users have the option to clone the source code directly from GitHub or pull the latest container image, allowing them to reproduce or expand the experiments locally with minimal setup effort. This workflow ensures scalability, reproducibility, and efficiency throughout the development process.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Georgia to merge::

hardest: deploying model in GCP
dvc in Bucket
setting up an environment to be shared across team
building yaml trigger workflow
building dockerfiles
wandb setup was easy
easiest: building the template with cookiecutter

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 31 fill here --- -Write what you did each person
Student s232253 was in charge of writing evaluate.py, visualize.py locally, and making dockerfiles for train.py and evaluate.py locally, and then profiling and logging. Then calculate the code coverage and add continues workflows.

