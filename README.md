luigiflow
=====

luigiflow is a simple machine learning task manager.
luigiflow is built on two popular Python frameworks: luigi and mlflow.
For more detail explanation about this framework, take a look at my blog post:
[Toward the minimalism of machine learning experiment workflows](https://blog.whiro.me/minimal-ml-experiment-workflows/)

## Get started!

### Install this package

If you use pip, run the following command.
```bash
$ pip install git+https://github.com/tarohi24/luigiflow.git
```

2. Launch an mlflow server. If you start a server on your local machine, use the following command.

```bash
#!/bin/bash 
DB_URI="sqlite:///db.sqlite3"  # Specify backend database
ARTIFACTS_DIR="/PATH/TO/ARTIFACTS_DIR"  # Specify a directory where mlflow saves arfifacts  
PORT=8000

mlflow server \
    --backend-store-uri ${DB_URI} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --default-artifact-root ${ARTIFACTS_DIR}
```

3. Implement a task. Let's take an example of `HelloTask`. `HelloTask `is a task just to print a parametrized message.

```python
import luigi
from luigiflow.task import MlflowTask


class HelloTask(MlflowTask):
   message: str = luigi.Parameter()

   @classmethod
   def get_experiment_name(cls):
      # Set the name of its MLflow experiment
      return "hello"

   @classmethod
   def get_artifact_filenames(cls):
      # {file_identifier: filename} for any mlflow artifacts that this task outputs
      return dict()

   def requires(self):
      # {task_name: task}
      return dict()

   def _run(self):
      # Specify the job contents
      print(self.message)
```

4. (necessary only if your task has parameters) Prepare a jsonnet file to set parameter values.
   Let's create "config.jsonnet", as follows.

```jsonnet
# config.jsonnet
local message = "good morning!";
{
  "HelloTask": {
    "message": message,
  }
}
```

5. Run a task using `luigiflow.run()`. For example, you can run the `HelloTask` as follows.

```python
import luigiflow

config_path = "config.jsonnet"
luigiflow.run(
  task_cls=HelloTask,
  mlflow_tracking_uri="http://localhost:8000",  # set your mlflow's uri 
  config_path=config_path,
  local_scheduler=True,
)
```
